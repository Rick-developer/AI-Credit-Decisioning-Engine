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


def test_save_and_load_model(sample_dataset, tmp_path):
    """Test that a trained model can be saved and loaded with identical predictions."""
    model = CreditRiskModel()
    model.train(sample_dataset)
    
    # Save
    model_path = model.save_model(output_dir=str(tmp_path), version="test_1.0")
    
    import json
    from pathlib import Path
    
    # Verify files exist
    assert Path(model_path).exists()
    manifest_path = tmp_path / "model_vtest_1.0_manifest.json"
    assert manifest_path.exists()
    
    # Verify manifest content
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["version"] == "test_1.0"
    assert manifest["algorithm"] == "XGBoost + CalibratedClassifierCV (Platt scaling)"
    assert manifest["training_metrics"] is not None
    assert "auc_roc" in manifest["training_metrics"]
    assert manifest["feature_names"] is not None
    assert len(manifest["protected_features_excluded"]) == 3
    
    # Load and verify identical predictions
    loaded_model = CreditRiskModel.load_model(model_path)
    single = sample_dataset.iloc[[0]].copy()
    
    original_decision = model.predict_with_explanation(single, threshold=0.45)
    loaded_decision = loaded_model.predict_with_explanation(single, threshold=0.45)
    
    assert abs(original_decision.risk_score - loaded_decision.risk_score) < 1e-6
    assert original_decision.approved == loaded_decision.approved

