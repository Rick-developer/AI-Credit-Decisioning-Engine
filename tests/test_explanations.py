import pytest
from src.data.models import Applicant, CreditDecision, AgeGroup, Gender, NationalityGroup, EmploymentType
from src.explanations.engine import ExplanationEngine

@pytest.fixture
def sample_applicant():
    return Applicant(
        applicant_id="APP-TEST-999",
        age=30,
        age_group=AgeGroup.EARLY_CAREER,
        gender=Gender.MALE,
        nationality=NationalityGroup.SOUTH_ASIAN,
        employment_type=EmploymentType.SALARIED,
        employment_tenure_months=24,
        monthly_income=5000.0,
        monthly_obligations=3000.0,
        monthly_spend=1500.0,
        existing_credit_exposure=10000.0,
        transaction_history_months=12,
        num_previous_applications=1,
        num_late_payments=2,
        requested_amount=2000.0,
        defaulted=1,
        requires_bureau_check=True,
        max_eligible_credit=15000.0
    )

@pytest.fixture
def declined_decision():
    return CreditDecision(
        applicant_id="APP-TEST-999",
        approved=False,
        risk_score=0.85,
        threshold_used=0.45,
        top_factors=[
            {"feature": "debt_to_income_ratio", "impact": 0.5, "value": 0.6},
            {"feature": "late_payment_frequency", "impact": 0.3, "value": 0.16},
            {"feature": "employment_tenure_months", "impact": -0.1, "value": 24.0}
        ]
    )

@pytest.fixture
def approved_decision():
    return CreditDecision(
        applicant_id="APP-TEST-999",
        approved=True,
        risk_score=0.15,
        threshold_used=0.45,
        top_factors=[
            {"feature": "debt_to_income_ratio", "impact": -0.5, "value": 0.1},
            {"feature": "late_payment_frequency", "impact": -0.3, "value": 0.0}
        ]
    )

def test_approved_decision_skips_llm(sample_applicant, approved_decision):
    """Test that an approved decision gets a standard message without hitting the LLM."""
    engine = ExplanationEngine(api_key="fake-key-wont-be-used")
    
    result = engine.generate_explanation(sample_applicant, approved_decision)
    
    assert result.adverse_action_notice == "Application approved. No adverse action taken."
    assert "APPROVED" in result.reason_codes

def test_mock_explanation_generation(sample_applicant, declined_decision):
    """Test that the engine falls back to mock mode when no API key is present."""
    # Initialize without API key
    engine = ExplanationEngine(api_key=None)
    
    result = engine.generate_explanation(sample_applicant, declined_decision)
    
    assert "[MOCK]" in result.adverse_action_notice
    assert "debt_to_income_ratio" in result.adverse_action_notice
    assert "DEBT_TO_INCOME_RATIO" in result.reason_codes

def test_build_prompt(sample_applicant, declined_decision):
    """Test that the prompt string is constructed correctly with SHAP factors."""
    engine = ExplanationEngine(api_key=None)
    
    prompt = engine._build_prompt(sample_applicant, declined_decision)
    
    # Should include applicant details
    assert str(sample_applicant.requested_amount) in prompt
    assert str(sample_applicant.monthly_income) in prompt
    
    # Should include ONLY positive SHAP impacts (risk-increasing factors)
    assert "debt_to_income_ratio" in prompt
    assert "late_payment_frequency" in prompt
    assert "employment_tenure_months" not in prompt # Impact was -0.1, so it decreased risk and shouldn't be in the prompt
