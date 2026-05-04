import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer risk signals from raw applicant data for the ML model.
    
    Creates domain-specific financial ratios and risk proxies that are 
    commonly used in BNPL/lending decisioning.
    """
    df_engineered = df.copy()
    
    # ── Income & Burden Ratios ──
    # Debt-to-Income (DTI) ratio - Critical for CBUAE responsible lending
    df_engineered['debt_to_income_ratio'] = df_engineered['monthly_obligations'] / df_engineered['monthly_income']
    
    # Spend-to-Income ratio - Indicates free cash flow
    df_engineered['spend_to_income_ratio'] = df_engineered['monthly_spend'] / df_engineered['monthly_income']
    
    # Requested amount relative to income
    df_engineered['requested_to_income_ratio'] = df_engineered['requested_amount'] / df_engineered['monthly_income']
    
    # Total future DTI (if approved)
    # Approximating monthly payment for BNPL as requested_amount / 4 (typical split-in-4 model)
    estimated_new_payment = df_engineered['requested_amount'] / 4.0
    df_engineered['post_approval_dti'] = (df_engineered['monthly_obligations'] + estimated_new_payment) / df_engineered['monthly_income']

    # ── Credit History Signals ──
    # Credit utilization proxy
    # We use max_eligible_credit as a proxy for their total credit limit
    df_engineered['credit_utilization_proxy'] = np.where(
        df_engineered['max_eligible_credit'] > 0,
        df_engineered['existing_credit_exposure'] / df_engineered['max_eligible_credit'],
        0.0
    )
    
    # Late payment frequency (per month of history)
    df_engineered['late_payment_frequency'] = df_engineered['num_late_payments'] / np.maximum(1, df_engineered['transaction_history_months'])
    
    # Clean up any potential inf/nan from division by zero
    df_engineered.replace([np.inf, -np.inf], 0.0, inplace=True)
    df_engineered.fillna(0.0, inplace=True)
    
    return df_engineered

def get_feature_definitions() -> dict[str, str]:
    """Return human-readable definitions of engineered features for SHAP/LLM grounding."""
    return {
        "debt_to_income_ratio": "Proportion of monthly income committed to existing debt obligations.",
        "spend_to_income_ratio": "Proportion of monthly income spent on general expenses.",
        "requested_to_income_ratio": "Requested BNPL amount relative to monthly income.",
        "post_approval_dti": "Estimated debt-to-income ratio if this application is approved.",
        "credit_utilization_proxy": "Estimated percentage of available credit currently in use.",
        "late_payment_frequency": "Historical rate of late payments relative to account age."
    }
