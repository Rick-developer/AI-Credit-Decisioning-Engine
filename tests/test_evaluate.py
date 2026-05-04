import pandas as pd
import pytest
from src.evaluation.evaluate import PipelineEvaluator
from src.models.scorer import CreditRiskModel
from src.data.generator import generate_applicant

@pytest.fixture
def test_dataset():
    """Generate a small sample dataset for testing the evaluator."""
    records = [generate_applicant(f"APP-EVAL-{str(i).zfill(3)}") for i in range(150)]
    return pd.DataFrame(records)

def test_pipeline_evaluator(test_dataset):
    """Test that the evaluator successfully runs and returns metrics."""
    model = CreditRiskModel()
    
    # Needs to be trained first to initialize the XGBoost model internals
    model.train(test_dataset)
    
    evaluator = PipelineEvaluator(model, test_dataset)
    
    results = evaluator.evaluate(threshold=0.45)
    
    # Check structure
    assert "ml_metrics" in results
    assert "fairness_metrics" in results
    assert "explanation_fidelity" in results
    
    # Check ML metrics
    assert "auc_roc" in results["ml_metrics"]
    
    # Check fairness metrics
    assert "passes_all_fairness" in results["fairness_metrics"]
    
    # Check explanation fidelity
    assert "grounding_rate" in results["explanation_fidelity"]
    assert 0.0 <= results["explanation_fidelity"]["grounding_rate"] <= 1.0
