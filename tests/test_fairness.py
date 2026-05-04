import pandas as pd
import pytest
import numpy as np
from src.fairness.auditor import FairnessAuditor

@pytest.fixture
def fairness_df():
    """Create a mock dataframe with intentional bias for testing."""
    # 100 records
    np.random.seed(42)
    df = pd.DataFrame({
        'applicant_id': [f"APP-{i}" for i in range(100)],
        'age_group': ['YOUNG'] * 50 + ['SENIOR'] * 50,
        'gender': ['MALE'] * 70 + ['FEMALE'] * 30,
        'nationality': ['EMIRATI'] * 20 + ['WESTERN'] * 80,
        # Intentionally biased risk scores:
        # YOUNG gets higher risk scores (worse) than SENIOR
        'risk_score': list(np.random.normal(0.6, 0.1, 50)) + list(np.random.normal(0.3, 0.1, 50)),
        # Intentional proxy variable: employment_tenure is highly correlated with age_group
        'employment_tenure_months': list(np.random.normal(12, 5, 50)) + list(np.random.normal(120, 20, 50))
    })
    return df

def test_demographic_parity(fairness_df):
    auditor = FairnessAuditor()
    reports = auditor.check_demographic_parity(fairness_df, threshold=0.45)
    
    # Should generate a report for each protected class
    assert len(reports) == 3
    
    # Find the age_group report
    age_report = next(r for r in reports if r.protected_attribute == 'age_group')
    
    # SENIOR should have much higher approval rate than YOUNG
    assert age_report.group_results['SENIOR'] > age_report.group_results['YOUNG']
    
    # Max disparity should be large (likely > 15%), so it should fail
    assert age_report.max_disparity > 0.15
    assert age_report.passes_threshold == False
    assert "FAILED" in age_report.narrative

def test_proxy_variable_detection(fairness_df):
    auditor = FairnessAuditor()
    proxies = auditor.detect_proxy_variables(fairness_df)
    
    # Should detect employment_tenure_months as a proxy for age_group
    assert len(proxies) > 0
    proxy_found = False
    for p in proxies:
        if p['protected_class'] == 'age_group' and p['proxy_feature'] == 'employment_tenure_months':
            proxy_found = True
            assert p['p_value'] < 0.001
            assert p['max_difference_pct'] > 0.2
            
    assert proxy_found
