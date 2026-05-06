import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import joblib
import json
from datetime import datetime, timezone
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV

from src.data.models import CreditDecision
from src.features.engineer import engineer_features

class CreditRiskModel:
    """XGBoost model for credit risk assessment with SHAP explanations.
    
    Demonstrates 'Fairness Through Unawareness' by explicitly dropping 
    protected demographic classes before training, setting up the 
    Fairness Auditor (Layer 3) to prove that this doesn't prevent proxy bias.
    """
    def __init__(self):
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            max_depth=4,
            learning_rate=0.05,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.explainer = None
        self.feature_names = None
        
        # Protected classes that MUST NOT be used for training
        self.protected_features = ['age_group', 'gender', 'nationality']
        # IDs, targets, and derived fields that leak target info or shouldn't be features
        # NOTE: max_eligible_credit is derived from monthly_income — keeping it
        #       would redundantly leak income information (audit finding C-3).
        self.meta_features = ['applicant_id', 'defaulted', 'requires_bureau_check', 'max_eligible_credit']
        
        self.categorical_features = ['employment_type']
        self.calibrated_model = None
        self._training_metrics = None
        self._version = None

    def _prepare_features(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Engineer features and drop protected/meta columns."""
        # 1. Engineer domain features
        df_eng = engineer_features(df)
        
        # 2. One-hot encode categoricals
        df_encoded = pd.get_dummies(df_eng, columns=self.categorical_features, drop_first=True)
        
        # 3. Drop protected classes and meta columns
        cols_to_drop = [c for c in self.protected_features + self.meta_features if c in df_encoded.columns]
        X = df_encoded.drop(columns=cols_to_drop)
        
        if is_training:
            self.feature_names = list(X.columns)
        
        # Ensure all columns are numeric
        cols = list(X.columns)
        for col in cols:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col])
                except Exception:
                    X = X.drop(columns=[col])
                    if is_training and col in self.feature_names:
                        self.feature_names.remove(col)
                        
        return X

    def train(self, df: pd.DataFrame) -> dict[str, float]:
        """Train the model and return evaluation metrics."""
        X = self._prepare_features(df, is_training=True)
        y = df['defaulted']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # H-2 FIX: Handle class imbalance via scale_pos_weight
        neg_count = int((y_train == 0).sum())
        pos_count = int((y_train == 1).sum())
        if pos_count > 0:
            self.model.set_params(scale_pos_weight=neg_count / pos_count)
        
        self.model.fit(X_train, y_train)
        
        # H-3 FIX: Platt scaling calibration for well-calibrated probabilities
        self.calibrated_model = CalibratedClassifierCV(
            self.model, cv=5, method='sigmoid'
        )
        self.calibrated_model.fit(X_train, y_train)
        
        # Initialize SHAP explainer (uses raw model for SHAP, calibrated for predictions)
        self.explainer = shap.TreeExplainer(self.model)
        
        # Evaluate using calibrated probabilities
        y_prob = self.calibrated_model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        
        # H-8 FIX: Add KS and Gini metrics
        ks_stat = self._compute_ks(y_test, y_prob)
        gini = 2 * float(roc_auc_score(y_test, y_prob)) - 1
        
        metrics = {
            "auc_roc": float(roc_auc_score(y_test, y_prob)),
            "gini": gini,
            "ks_statistic": ks_stat,
            "brier_score": float(brier_score_loss(y_test, y_prob)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "class_balance": f"{pos_count}/{neg_count} (default/non-default)"
        }
        self._training_metrics = metrics
        return metrics
    
    @staticmethod
    def _compute_ks(y_true, y_prob) -> float:
        """Compute Kolmogorov-Smirnov statistic for model discrimination."""
        from scipy.stats import ks_2samp
        prob_default = y_prob[y_true == 1]
        prob_non_default = y_prob[y_true == 0]
        if len(prob_default) == 0 or len(prob_non_default) == 0:
            return 0.0
        stat, _ = ks_2samp(prob_default, prob_non_default)
        return float(stat)

    def predict_with_explanation(self, df_applicant: pd.DataFrame, threshold: float = 0.45) -> CreditDecision:
        """Predict risk for a single applicant and generate SHAP attributions."""
        if len(df_applicant) != 1:
            raise ValueError("Expected exactly 1 applicant record")
            
        applicant_id = df_applicant['applicant_id'].iloc[0]
        
        X = self._prepare_features(df_applicant)
        
        # Make sure columns match training
        if self.feature_names:
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.feature_names]
            
        # Predict using calibrated model if available, raw model otherwise
        predictor = self.calibrated_model if self.calibrated_model else self.model
        prob_default = float(predictor.predict_proba(X)[0, 1])
        approved = prob_default <= threshold
        
        # SHAP explanation
        shap_values = self.explainer.shap_values(X)[0]
        
        # Get top 3 risk-increasing factors (positive SHAP values)
        # and top 2 risk-decreasing factors (negative SHAP values)
        feature_impacts = [{"feature": feat, "impact": float(val), "value": float(X[feat].iloc[0])} 
                           for feat, val in zip(X.columns, shap_values)]
        
        # Sort by absolute impact for the top overall factors
        feature_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
        top_factors = feature_impacts[:5]
        
        return CreditDecision(
            applicant_id=applicant_id,
            approved=approved,
            risk_score=prob_default,
            threshold_used=threshold,
            top_factors=top_factors,
            adverse_action_notice=None,
            reason_codes=[],
            regulatory_citations=[]
        )

    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run predictions on a batch of applicants for fairness testing."""
        X = self._prepare_features(df)
        predictor = self.calibrated_model if self.calibrated_model else self.model
        probs = predictor.predict_proba(X)[:, 1]
        
        result_df = df.copy()
        result_df['risk_score'] = probs
        return result_df

    def save_model(self, output_dir: str = "data/models", version: str = None) -> str:
        """Serialize the trained model, calibrator, and metadata to disk.
        
        Saves:
        - model_v{version}.joblib: XGBoost model + calibrated model + SHAP explainer
        - model_v{version}_manifest.json: Version manifest with hyperparameters,
          training metrics, feature names, and timestamp
        
        Args:
            output_dir: Directory to save model artifacts.
            version: Version string (e.g., '1.0'). Auto-generated if not provided.
            
        Returns:
            Path to the saved model file.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before saving.")
        
        if version is None:
            version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        self._version = version
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model artifacts
        model_file = output_path / f"model_v{version}.joblib"
        artifact = {
            "model": self.model,
            "calibrated_model": self.calibrated_model,
            "feature_names": self.feature_names,
            "protected_features": self.protected_features,
            "meta_features": self.meta_features,
        }
        joblib.dump(artifact, model_file)
        
        # Save version manifest
        manifest = {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "algorithm": "XGBoost + CalibratedClassifierCV (Platt scaling)",
            "hyperparameters": {
                "max_depth": self.model.get_params().get("max_depth"),
                "learning_rate": self.model.get_params().get("learning_rate"),
                "n_estimators": self.model.get_params().get("n_estimators"),
                "subsample": self.model.get_params().get("subsample"),
                "colsample_bytree": self.model.get_params().get("colsample_bytree"),
                "scale_pos_weight": self.model.get_params().get("scale_pos_weight"),
            },
            "feature_names": self.feature_names,
            "protected_features_excluded": self.protected_features,
            "training_metrics": self._training_metrics,
        }
        manifest_file = output_path / f"model_v{version}_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
        
        return str(model_file)
    
    @classmethod
    def load_model(cls, model_path: str) -> "CreditRiskModel":
        """Load a previously saved model from disk.
        
        Args:
            model_path: Path to the .joblib model file.
            
        Returns:
            A CreditRiskModel instance with the loaded model.
        """
        artifact = joblib.load(model_path)
        
        instance = cls()
        instance.model = artifact["model"]
        instance.calibrated_model = artifact["calibrated_model"]
        instance.feature_names = artifact["feature_names"]
        instance.protected_features = artifact["protected_features"]
        instance.meta_features = artifact["meta_features"]
        instance.explainer = shap.TreeExplainer(instance.model)
        
        return instance
