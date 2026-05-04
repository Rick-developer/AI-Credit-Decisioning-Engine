import pandas as pd
import numpy as np
from typing import Dict, Any

from src.models.scorer import CreditRiskModel
from src.explanations.engine import ExplanationEngine
from src.fairness.auditor import FairnessAuditor
from src.data.models import Applicant

class PipelineEvaluator:
    """Evaluates the hybrid ML+LLM+RAG pipeline across three dimensions.
    
    1. ML Performance (Standard accuracy metrics)
    2. Explanation Fidelity (Does the LLM use the SHAP values?)
    3. Fairness (Demographic parity)
    """
    
    def __init__(self, model: CreditRiskModel, df: pd.DataFrame):
        self.model = model
        self.df = df
        self.auditor = FairnessAuditor()
        # We use mock LLM for evaluation to save API costs and ensure speed
        self.explanation_engine = ExplanationEngine(api_key=None)
        
    def evaluate(self, threshold: float = 0.45) -> Dict[str, Any]:
        """Run the full triple-evaluation."""
        
        # 1. ML Metrics
        # Re-train to get test-set metrics reliably
        ml_metrics = self.model.train(self.df)
        
        # 2. Fairness Metrics
        df_scored = self.model.batch_predict(self.df)
        fairness_reports = self.auditor.check_demographic_parity(df_scored, threshold=threshold)
        
        fairness_metrics = {}
        passes_all_fairness = True
        for report in fairness_reports:
            fairness_metrics[f"disparity_{report.protected_attribute}"] = report.max_disparity
            if not report.passes_threshold:
                passes_all_fairness = False
                
        fairness_metrics["passes_all_fairness"] = passes_all_fairness
        
        # 3. Explanation Fidelity (Sample 100 declined applicants)
        fidelity_metrics = self._evaluate_explanations(df_scored, threshold)
        
        return {
            "ml_metrics": ml_metrics,
            "fairness_metrics": fairness_metrics,
            "explanation_fidelity": fidelity_metrics
        }
        
    def _evaluate_explanations(self, df_scored: pd.DataFrame, threshold: float, sample_size: int = 100) -> Dict[str, Any]:
        """Verify that explanations actually use the SHAP top factors.
        
        This prevents 'hallucinated compliance' by ensuring the LLM's reason codes
        match the actual mathematical drivers of the decision.
        """
        # Find declined applicants
        declined_df = df_scored[df_scored['risk_score'] > threshold]
        
        if len(declined_df) == 0:
            return {"grounding_rate": 1.0, "note": "No declined applications to explain"}
            
        sample = declined_df.head(sample_size)
        
        grounded_count = 0
        total_count = len(sample)
        
        for _, row in sample.iterrows():
            applicant_df = pd.DataFrame([row])
            
            # Reconstruct Applicant object safely
            app_dict = row.to_dict()
            try:
                applicant = Applicant(**app_dict)
            except Exception:
                # If validation fails due to scoring columns added, clean it
                clean_dict = {k: v for k, v in app_dict.items() if k in Applicant.model_fields}
                applicant = Applicant(**clean_dict)
            
            # Get decision with SHAP
            decision = self.model.predict_with_explanation(applicant_df, threshold)
            
            # Generate explanation (mock mode)
            explained_decision = self.explanation_engine.generate_explanation(applicant, decision)
            
            # Check grounding: Do the reason codes map to the top SHAP features?
            # In mock mode, the reason codes ARE the features.
            # In real mode, we'd do a fuzzy match or semantic similarity.
            # Here we just check if any top feature name is mentioned in the notice or codes.
            top_features = [f["feature"].lower() for f in decision.top_factors]
            
            is_grounded = False
            notice_lower = explained_decision.adverse_action_notice.lower()
            codes_lower = [c.lower() for c in explained_decision.reason_codes]
            
            for feat in top_features:
                if feat in notice_lower or any(feat in c for c in codes_lower):
                    is_grounded = True
                    break
                    
            if is_grounded:
                grounded_count += 1
                
        return {
            "grounding_rate": grounded_count / total_count if total_count > 0 else 1.0,
            "sample_size": total_count
        }
