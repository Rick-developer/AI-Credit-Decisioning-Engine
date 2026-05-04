import pandas as pd
import numpy as np
from src.features.engineer import engineer_features, get_feature_definitions

def test_engineer_features_calculations():
    """Test that financial ratios are calculated correctly."""
    data = {
        "monthly_income": [10000.0, 5000.0],
        "monthly_obligations": [2000.0, 0.0],
        "monthly_spend": [3000.0, 4000.0],
        "requested_amount": [2000.0, 1000.0],
        "existing_credit_exposure": [5000.0, 0.0],
        "max_eligible_credit": [20000.0, 15000.0],
        "num_late_payments": [2, 0],
        "transaction_history_months": [10, 0] # Test division by zero
    }
    df = pd.DataFrame(data)
    
    engineered_df = engineer_features(df)
    
    # Test DTI
    assert np.isclose(engineered_df.loc[0, 'debt_to_income_ratio'], 0.2)
    assert np.isclose(engineered_df.loc[1, 'debt_to_income_ratio'], 0.0)
    
    # Test Spend to Income
    assert np.isclose(engineered_df.loc[0, 'spend_to_income_ratio'], 0.3)
    assert np.isclose(engineered_df.loc[1, 'spend_to_income_ratio'], 0.8)
    
    # Test post approval DTI
    # 2000 + (2000/4) = 2500 -> 2500/10000 = 0.25
    assert np.isclose(engineered_df.loc[0, 'post_approval_dti'], 0.25)
    
    # Test utilization
    assert np.isclose(engineered_df.loc[0, 'credit_utilization_proxy'], 0.25)
    assert np.isclose(engineered_df.loc[1, 'credit_utilization_proxy'], 0.0)
    
    # Test late payment frequency (division by zero handling)
    assert np.isclose(engineered_df.loc[0, 'late_payment_frequency'], 0.2)
    assert np.isclose(engineered_df.loc[1, 'late_payment_frequency'], 0.0)

def test_get_feature_definitions():
    """Test that dictionary of definitions is available."""
    defs = get_feature_definitions()
    assert "debt_to_income_ratio" in defs
    assert "post_approval_dti" in defs
