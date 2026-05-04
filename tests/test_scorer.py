import pandas as pd
import pytest
from src.models.scorer import CreditRiskModel
from src.data.generator import generate_applicant

@pytest.fixture
def sample_dataset():
    """Generate a small sample dataset for testing."""
    records = [generate_applicant(f"APP-TEST-{str(i).zfill(3)}") for i in range(100)]
    return pd.DataFrame(records)

def test_model_training(sample_dataset):
    """Test that the model trains and returns valid metrics."""
    model = CreditRiskModel()
    metrics = model.train(sample_dataset)
    
    assert "auc_roc" in metrics
    assert "brier_score" in metrics
    assert metrics["auc_roc"] > 0.0  # Just verify it calculates successfully
    assert model.explainer is not None

def test_single_prediction_with_explanation(sample_dataset):
    """Test predicting a single applicant returns CreditDecision with SHAP."""
    model = CreditRiskModel()
    model.train(sample_dataset)
    
    single_applicant = sample_dataset.iloc[[0]].copy()
    decision = model.predict_with_explanation(single_applicant, threshold=0.45)
    
    assert decision.applicant_id == single_applicant['applicant_id'].iloc[0]
    assert 0 <= decision.risk_score <= 1.0
    assert isinstance(decision.approved, bool)
    assert len(decision.top_factors) <= 5
    assert "feature" in decision.top_factors[0]
    assert "impact" in decision.top_factors[0]

def test_protected_classes_dropped(sample_dataset):
    """Test that the model does not train on protected classes."""
    model = CreditRiskModel()
    X = model._prepare_features(sample_dataset)
    
    assert 'gender' not in X.columns
    assert 'nationality' not in X.columns
    assert 'age_group' not in X.columns
    assert 'applicant_id' not in X.columns
    assert 'defaulted' not in X.columns
