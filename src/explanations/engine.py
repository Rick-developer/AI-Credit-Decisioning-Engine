import json
import os
from typing import Dict, Any, List

from groq import Groq
from pydantic import BaseModel

from src.data.models import Applicant, CreditDecision
from src.features.engineer import get_feature_definitions

class ExplanationResponse(BaseModel):
    """Structured output expected from the LLM."""
    adverse_action_notice: str
    reason_codes: List[str]

class ExplanationEngine:
    """LLM-powered explanation engine for credit decisions.
    
    Translates SHAP feature attributions into natural language 
    adverse action notices using Groq (Llama 3).
    
    CRITICAL PM DECISION: The LLM never makes the credit decision.
    It ONLY translates the ML model's SHAP values into compliance text.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the Groq client."""
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            # We'll mock the response if no API key is provided during testing
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
            
        self.feature_definitions = get_feature_definitions()

    def generate_explanation(self, applicant: Applicant, decision: CreditDecision) -> CreditDecision:
        """Generate an adverse action notice based on SHAP values."""
        
        # If approved, we don't need an adverse action notice
        if decision.approved:
            decision.adverse_action_notice = "Application approved. No adverse action taken."
            decision.reason_codes = ["APPROVED"]
            return decision
            
        # If no SHAP factors, we can't explain it
        if not decision.top_factors:
            decision.adverse_action_notice = "Application declined. Specific reasons could not be extracted."
            decision.reason_codes = ["UNKNOWN_REASON"]
            return decision

        prompt = self._build_prompt(applicant, decision)
        
        if self.client:
            try:
                # Use Groq to generate structured JSON output
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert UAE credit compliance officer. "
                                "Your job is to write a clear, professional adverse action notice "
                                "explaining why a BNPL loan was declined. You MUST base your "
                                "explanation strictly on the provided risk factors. Do not hallucinate."
                                "Return JSON matching the schema."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model="llama3-8b-8192", # Fast and capable enough for this
                    temperature=0.1, # Low temperature for consistency
                    response_format={"type": "json_object"}
                )
                
                response_str = chat_completion.choices[0].message.content
                response_data = json.loads(response_str)
                
                # Update the decision object
                decision.adverse_action_notice = response_data.get("adverse_action_notice", "Error generating notice.")
                decision.reason_codes = response_data.get("reason_codes", [])
                
            except Exception as e:
                decision.adverse_action_notice = f"Error generating explanation via LLM: {str(e)}"
                decision.reason_codes = ["LLM_ERROR"]
        else:
            # Mock mode for testing without API key
            factors_text = ", ".join([f["feature"] for f in decision.top_factors[:2]])
            decision.adverse_action_notice = f"[MOCK] Your application was declined primarily due to: {factors_text}."
            decision.reason_codes = [f["feature"].upper() for f in decision.top_factors[:2]]
            
        return decision

    def _build_prompt(self, applicant: Applicant, decision: CreditDecision) -> str:
        """Construct the prompt instructing the LLM to translate SHAP into compliance text."""
        
        # Map the raw SHAP feature names to human-readable definitions
        factors_context = []
        for factor in decision.top_factors:
            feat_name = factor["feature"]
            impact = factor["impact"]
            value = factor["value"]
            
            # Only include factors that increased risk (positive SHAP)
            if impact > 0:
                definition = self.feature_definitions.get(feat_name, "Raw risk signal")
                factors_context.append(
                    f"- {feat_name} (Value: {value:.2f}): {definition} (Impact magnitude: {impact:.4f})"
                )
        
        factors_text = "\n".join(factors_context)
        
        prompt = f"""
Please generate an adverse action notice explaining why this applicant was declined for a UAE BNPL loan.

Applicant Profile:
- Requested Amount: {applicant.requested_amount} AED
- Monthly Income: {applicant.monthly_income} AED

Key Factors Driving the Denial (from ML SHAP explainer):
{factors_text}

Instructions:
1. Write a professional, empathetic adverse action notice (max 3 sentences).
2. Explicitly mention the primary driving factors using clear, consumer-friendly language (do NOT use variable names like 'debt_to_income_ratio' directly, explain what it means).
3. Generate 1-3 short reason codes (e.g., 'HIGH_DEBT_BURDEN', 'INSUFFICIENT_HISTORY').

Return a JSON object with this exact structure:
{{
    "adverse_action_notice": "Your application was declined because...",
    "reason_codes": ["CODE_1", "CODE_2"]
}}
"""
        return prompt
