import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from src.data.models import AgeGroup, Applicant, EmploymentType, Gender, NationalityGroup

# Setting a seed for reproducibility
random.seed(42)

def generate_applicant(applicant_id: str) -> dict[str, Any]:
    """Generate a single realistic UAE BNPL applicant record."""
    
    # ── Demographics ──
    age = random.randint(18, 65)
    
    # Age group calculation matches the model
    if age <= 25:
        age_group = AgeGroup.YOUNG
    elif age <= 35:
        age_group = AgeGroup.EARLY_CAREER
    elif age <= 45:
        age_group = AgeGroup.MID_CAREER
    else:
        age_group = AgeGroup.SENIOR

    gender = random.choices(list(Gender), weights=[0.65, 0.35])[0]  # UAE skew
    
    # UAE Demographic approximation
    nationality = random.choices(
        list(NationalityGroup), 
        weights=[0.12, 0.45, 0.15, 0.15, 0.08, 0.05]
    )[0]

    # ── Employment & Income ──
    employment_type = random.choices(
        list(EmploymentType), 
        weights=[0.75, 0.15, 0.08, 0.02]
    )[0]
    
    # Injecting correlation between age and employment tenure (Proxy variable for bias)
    base_tenure = max(0, (age - 22) * 12)
    employment_tenure_months = int(random.gauss(base_tenure * 0.4, 12))
    if employment_tenure_months < 0 or employment_type == EmploymentType.UNEMPLOYED:
        employment_tenure_months = 0

    # Income based on employment and age/nationality (Realistic biases)
    if employment_type == EmploymentType.UNEMPLOYED:
        monthly_income = random.uniform(2000, 5000) # Maybe some allowance
    else:
        base_income = random.lognormvariate(9.0, 0.6) # Lognormal distribution for income
        
        # Nationality multipliers (Reflecting real-world disparities often found in data)
        if nationality == NationalityGroup.WESTERN:
            base_income *= 1.5
        elif nationality == NationalityGroup.SOUTH_ASIAN:
            base_income *= 0.8
            
        # Age multiplier
        if age_group == AgeGroup.SENIOR:
            base_income *= 1.3
            
        monthly_income = round(base_income, 2)
        
    # Ensure minimum income
    monthly_income = max(3000.0, monthly_income)

    # ── Financial Profile ──
    # Debt to income ratio
    dti_ratio = random.betavariate(2, 5) * 0.6 # Most people have 0-40% DTI
    monthly_obligations = round(monthly_income * dti_ratio, 2)
    
    # Spending habits
    monthly_spend = round(monthly_income * random.uniform(0.3, 0.8), 2)
    
    # Existing credit
    existing_credit_exposure = round(monthly_obligations * random.uniform(10, 40), 2)

    # ── Transaction History ──
    transaction_history_months = int(random.uniform(0, 48))
    num_previous_applications = random.randint(0, 5)
    
    # Late payments correlated with DTI and lower income
    late_payment_prob = (dti_ratio * 0.5) + (3000 / monthly_income * 0.2)
    num_late_payments = 0
    if random.random() < late_payment_prob:
        num_late_payments = random.randint(1, 3)

    # ── Request ──
    # Requesting between 500 and 20000
    requested_amount = round(random.uniform(500, min(20000, monthly_income * 3)), 2)

    # ── Default Logic (Hidden true function) ──
    # This determines the actual default value. 
    # We'll make it depend on DTI, late payments, and employment type.
    risk_score = 0.0
    risk_score += (dti_ratio * 0.4)
    risk_score += (num_late_payments * 0.15)
    if employment_type in [EmploymentType.FREELANCER, EmploymentType.UNEMPLOYED]:
        risk_score += 0.15
    if requested_amount > monthly_income:
        risk_score += 0.1
    if transaction_history_months < 6:
        risk_score += 0.05
        
    # Introduce some noise
    risk_score += random.gauss(0, 0.1)
    
    # Default threshold (adjusted for ~15% default rate)
    defaulted = 1 if risk_score > 0.45 else 0

    # ── Derived Fields ──
    requires_bureau_check = existing_credit_exposure + requested_amount > 5000
    max_eligible_credit = min(20000.0, monthly_income * 3)

    # Construct the raw dictionary
    raw_data = {
        "applicant_id": applicant_id,
        "age": age,
        "age_group": age_group,
        "gender": gender,
        "nationality": nationality,
        "employment_type": employment_type,
        "employment_tenure_months": employment_tenure_months,
        "monthly_income": monthly_income,
        "monthly_obligations": monthly_obligations,
        "monthly_spend": monthly_spend,
        "existing_credit_exposure": existing_credit_exposure,
        "transaction_history_months": transaction_history_months,
        "num_previous_applications": num_previous_applications,
        "num_late_payments": num_late_payments,
        "requested_amount": requested_amount,
        "defaulted": defaulted,
        "requires_bureau_check": requires_bureau_check,
        "max_eligible_credit": max_eligible_credit
    }
    
    # Validate via Pydantic model
    applicant = Applicant(**raw_data)
    
    return applicant.model_dump()


def generate_dataset(num_records: int, output_path: str):
    """Generate a dataset of applicants and save to CSV."""
    print(f"Generating {num_records} applicant records...")
    
    records = []
    default_count = 0
    
    for i in range(num_records):
        applicant_id = f"APP-{str(i+1).zfill(6)}"
        try:
            record = generate_applicant(applicant_id)
            records.append(record)
            if record["defaulted"] == 1:
                default_count += 1
        except Exception as e:
            print(f"Validation error for applicant {applicant_id}: {e}")
            
    # Save to CSV
    if not records:
        print("No valid records generated.")
        return
        
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = list(records[0].keys())
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
        
    print(f"Dataset generated successfully at {output_path}")
    print(f"Total records: {len(records)}")
    print(f"Default rate: {default_count / len(records):.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic UAE BNPL applicant data.")
    parser.add_argument("--count", type=int, default=5000, help="Number of records to generate")
    parser.add_argument("--output", type=str, default="data/sample/applicants.csv", help="Output CSV path")
    args = parser.parse_args()
    
    generate_dataset(args.count, args.output)
