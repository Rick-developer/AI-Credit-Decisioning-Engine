"""Data models for credit applicants and decisions.

Uses Pydantic for structured data validation per CLAUDE.md standards.
All monetary values in AED (UAE Dirham).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any

from pydantic import BaseModel, Field, field_validator, model_validator


class Gender(str, Enum):
    """Applicant gender — protected class for fairness testing."""
    MALE = "male"
    FEMALE = "female"


class NationalityGroup(str, Enum):
    """Nationality grouping — protected class for fairness testing.

    UAE demographics: ~88% expatriates, ~12% Emirati citizens.
    Grouped to avoid sparse categories while preserving fairness signal.
    """
    EMIRATI = "emirati"
    SOUTH_ASIAN = "south_asian"        # India, Pakistan, Bangladesh, Sri Lanka
    SOUTHEAST_ASIAN = "southeast_asian"  # Philippines, Indonesia
    ARAB_EXPAT = "arab_expat"          # Egypt, Jordan, Lebanon, Syria
    WESTERN = "western"                # US, UK, EU, Australia
    OTHER = "other"


class AgeGroup(str, Enum):
    """Age bracket — protected class for fairness testing."""
    YOUNG = "18-25"
    EARLY_CAREER = "26-35"
    MID_CAREER = "36-45"
    SENIOR = "46-60"


class EmploymentType(str, Enum):
    """Employment status categories."""
    SALARIED = "salaried"
    SELF_EMPLOYED = "self_employed"
    FREELANCER = "freelancer"
    UNEMPLOYED = "unemployed"


class Applicant(BaseModel):
    """A UAE BNPL/lending credit applicant.

    All monetary fields are in AED.
    CBUAE constraints:
        - Max credit: AED 20,000 or 3 months' net income (whichever is lower)
        - Bureau check required if total exposure > AED 5,000
    """
    # ── Identifiers ──
    applicant_id: str = Field(description="Unique applicant identifier")

    # ── Demographics (Protected Classes) ──
    age: int = Field(ge=18, le=65, description="Applicant age in years")
    age_group: Optional[AgeGroup] = Field(default=None, description="Age bracket for fairness analysis")
    gender: Gender = Field(description="Applicant gender")
    nationality: NationalityGroup = Field(description="Nationality group")

    # ── Employment ──
    employment_type: EmploymentType = Field(description="Employment status")
    employment_tenure_months: int = Field(ge=0, description="Months at current employer")

    # ── Financial Profile (AED) ──
    monthly_income: float = Field(gt=0, description="Monthly net income in AED")
    monthly_obligations: float = Field(ge=0, description="Existing monthly debt payments in AED")
    monthly_spend: float = Field(ge=0, description="Average monthly spending in AED")
    existing_credit_exposure: float = Field(ge=0, description="Total existing credit in AED")

    # ── Transaction History ──
    transaction_history_months: int = Field(ge=0, description="Months of transaction history available")
    num_previous_applications: int = Field(ge=0, description="Previous BNPL/credit applications")
    num_late_payments: int = Field(ge=0, description="Historical late payment count")

    # ── Request ──
    requested_amount: float = Field(gt=0, le=20_000, description="Requested BNPL amount in AED")

    # ── Target Variable ──
    defaulted: int = Field(ge=0, le=1, description="1 if applicant defaulted, 0 otherwise")

    # ── Derived (for CBUAE compliance) ──
    requires_bureau_check: bool = Field(
        description="True if total exposure > AED 5,000 (CBUAE mandate)"
    )
    max_eligible_credit: float = Field(
        description="Min(AED 20,000, 3 × monthly_income) per CBUAE"
    )

    @model_validator(mode="before")
    @classmethod
    def compute_age_group(cls, data: Any) -> Any:
        """Auto-compute age group from age if not provided."""
        if isinstance(data, dict):
            if not data.get("age_group"):
                age = data.get("age", 30)
                if age <= 25:
                    data["age_group"] = AgeGroup.YOUNG
                elif age <= 35:
                    data["age_group"] = AgeGroup.EARLY_CAREER
                elif age <= 45:
                    data["age_group"] = AgeGroup.MID_CAREER
                else:
                    data["age_group"] = AgeGroup.SENIOR
        return data

    model_config = {"frozen": False}


class CreditDecision(BaseModel):
    """The output of the credit decisioning pipeline."""
    applicant_id: str
    approved: bool
    risk_score: float = Field(ge=0, le=1, description="P(default) from ML model")
    threshold_used: float = Field(ge=0, le=1, description="Approval threshold")

    # ── SHAP-based explanation ──
    top_factors: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top contributing features with SHAP values"
    )

    # ── LLM-generated explanation ──
    adverse_action_notice: Optional[str] = Field(
        default=None,
        description="Natural language explanation for denial (Layer 2 output)"
    )
    reason_codes: list[str] = Field(
        default_factory=list,
        description="Structured reason codes for the decision"
    )

    # ── Regulatory grounding ──
    regulatory_citations: list[str] = Field(
        default_factory=list,
        description="CBUAE regulation references supporting the explanation"
    )


@dataclass
class FairnessReport:
    """Output of the fairness audit module."""
    metric_name: str
    protected_attribute: str
    group_results: dict[str, float] = field(default_factory=dict)
    max_disparity: float = 0.0
    passes_threshold: bool = True
    narrative: str = ""
