import pytest
from src.data.models import AgeGroup, Applicant, EmploymentType, Gender, NationalityGroup
from src.data.generator import generate_applicant

def test_generate_applicant_validity():
    """Test that generated applicants pass Pydantic validation."""
    record = generate_applicant("APP-TEST-1")
    
    # This will raise ValidationError if the record is invalid
    applicant = Applicant(**record)
    
    assert applicant.applicant_id == "APP-TEST-1"
    assert applicant.age >= 18
    assert applicant.monthly_income >= 3000
    assert 0 <= applicant.defaulted <= 1
    assert applicant.requested_amount <= 20000

def test_cbuau_constraints_enforced():
    """Test that CBUAE constraints are correctly computed."""
    record = generate_applicant("APP-TEST-2")
    applicant = Applicant(**record)
    
    # Max eligible credit must be min(20000, 3 * monthly_income)
    expected_max_credit = min(20000.0, applicant.monthly_income * 3)
    assert applicant.max_eligible_credit == expected_max_credit
    
    # Bureau check required if total exposure > 5000
    expected_bureau_check = (applicant.existing_credit_exposure + applicant.requested_amount) > 5000
    assert applicant.requires_bureau_check == expected_bureau_check

def test_age_group_computation():
    """Test that age_group is computed correctly based on age."""
    # We can test this directly on the Pydantic model
    raw_data = {
        "applicant_id": "APP-TEST-3",
        "age": 22,
        "gender": Gender.FEMALE,
        "nationality": NationalityGroup.SOUTH_ASIAN,
        "employment_type": EmploymentType.SALARIED,
        "employment_tenure_months": 12,
        "monthly_income": 4000,
        "monthly_obligations": 1000,
        "monthly_spend": 2000,
        "existing_credit_exposure": 0,
        "transaction_history_months": 12,
        "num_previous_applications": 0,
        "num_late_payments": 0,
        "requested_amount": 1000,
        "defaulted": 0,
        "requires_bureau_check": False,
        "max_eligible_credit": 12000,
        # Intentionally missing age_group to test auto-computation
    }
    
    # Test age 22 -> YOUNG
    app1 = Applicant(**raw_data)
    assert app1.age_group == AgeGroup.YOUNG
    
    # Test age 30 -> EARLY_CAREER
    raw_data["age"] = 30
    app2 = Applicant(**raw_data)
    assert app2.age_group == AgeGroup.EARLY_CAREER
    
    # Test age 40 -> MID_CAREER
    raw_data["age"] = 40
    app3 = Applicant(**raw_data)
    assert app3.age_group == AgeGroup.MID_CAREER
    
    # Test age 50 -> SENIOR
    raw_data["age"] = 50
    app4 = Applicant(**raw_data)
    assert app4.age_group == AgeGroup.SENIOR
