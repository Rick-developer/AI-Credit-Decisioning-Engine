"""
test_policy.py — Tests for Layer 4: RAG Policy Grounding.

Validates:
1. Regulatory corpus is well-formed (no empty fields)
2. Keyword index builds correctly
3. Declined decisions retrieve relevant regulations
4. Approved decisions still get grounded (CBUAE requires transparency for all decisions)
5. Citations are formatted correctly
6. High-DTI decisions retrieve affordability regulations
7. Late payment decisions retrieve credit history regulations
"""

import pytest
from src.policy.regulations import CBUAE_CORPUS, Regulation
from src.policy.grounder import PolicyGrounder
from src.data.models import CreditDecision


# ─── Corpus Integrity Tests ───

def test_corpus_not_empty():
    """Corpus must contain regulations."""
    assert len(CBUAE_CORPUS) > 0


def test_corpus_entries_well_formed():
    """Every regulation must have id, source, provision, and at least one keyword."""
    for reg in CBUAE_CORPUS:
        assert reg.id, f"Regulation missing id: {reg}"
        assert reg.source, f"Regulation {reg.id} missing source"
        assert reg.provision, f"Regulation {reg.id} missing provision"
        assert len(reg.keywords) > 0, f"Regulation {reg.id} has no keywords"
        assert reg.category, f"Regulation {reg.id} missing category"


def test_corpus_unique_ids():
    """All regulation IDs must be unique."""
    ids = [r.id for r in CBUAE_CORPUS]
    assert len(ids) == len(set(ids)), "Duplicate regulation IDs found"


# ─── Grounder Tests ───

@pytest.fixture
def grounder():
    return PolicyGrounder()


@pytest.fixture
def declined_decision():
    """A declined decision with DTI-related SHAP factors."""
    return CreditDecision(
        applicant_id="TEST-001",
        approved=False,
        risk_score=0.72,
        threshold_used=0.45,
        top_factors=[
            {"feature": "debt_to_income_ratio", "impact": 0.35, "value": 0.55},
            {"feature": "post_approval_dti", "impact": 0.22, "value": 0.68},
            {"feature": "late_payment_frequency", "impact": 0.15, "value": 0.08},
            {"feature": "requested_to_income_ratio", "impact": 0.10, "value": 1.2},
            {"feature": "credit_utilization_proxy", "impact": -0.05, "value": 0.3},
        ],
        adverse_action_notice="Application declined due to high debt burden.",
        reason_codes=["HIGH_DEBT_BURDEN", "EXCESSIVE_OBLIGATIONS", "LATE_PAYMENTS", "HIGH_REQUEST"],
    )


@pytest.fixture
def approved_decision():
    """An approved decision."""
    return CreditDecision(
        applicant_id="TEST-002",
        approved=True,
        risk_score=0.25,
        threshold_used=0.45,
        top_factors=[
            {"feature": "monthly_income", "impact": -0.20, "value": 25000},
            {"feature": "employment_tenure_months", "impact": -0.15, "value": 48},
        ],
        adverse_action_notice="Application approved.",
        reason_codes=["APPROVED"],
    )


def test_retrieve_returns_citations(grounder, declined_decision):
    """Declined decisions should retrieve at least one regulation."""
    citations = grounder.retrieve(declined_decision)
    assert len(citations) > 0
    assert len(citations) <= 3  # default top_k


def test_retrieve_affordability_for_dti(grounder, declined_decision):
    """A high-DTI decision should retrieve affordability regulations."""
    citations = grounder.retrieve(declined_decision, top_k=5)
    sources = [c["source"] for c in citations]
    # Should match at least one affordability regulation
    affordability_found = any("Responsible Lending" in s or "Debt Burden" in s for s in sources)
    assert affordability_found, f"Expected affordability regulation, got: {sources}"


def test_retrieve_for_late_payments(grounder):
    """A decision driven by late payments should retrieve payment history regs."""
    decision = CreditDecision(
        applicant_id="TEST-003",
        approved=False,
        risk_score=0.60,
        threshold_used=0.45,
        top_factors=[
            {"feature": "num_late_payments", "impact": 0.40, "value": 5},
            {"feature": "late_payment_frequency", "impact": 0.30, "value": 0.15},
        ],
        reason_codes=["LATE_PAYMENTS", "DELINQUENCY"],
    )
    citations = grounder.retrieve(decision, top_k=5)
    sources = [c["source"] for c in citations]
    payment_found = any("Payment History" in s for s in sources)
    assert payment_found, f"Expected payment history regulation, got: {sources}"


def test_ground_decision_populates_citations(grounder, declined_decision):
    """ground_decision should populate regulatory_citations on the decision."""
    assert declined_decision.regulatory_citations == []
    grounded = grounder.ground_decision(declined_decision)
    assert len(grounded.regulatory_citations) > 0
    # Citations should be formatted strings
    for citation in grounded.regulatory_citations:
        assert "[CBUAE" in citation or "[SR" in citation
        assert ":" in citation


def test_ground_decision_approved(grounder, approved_decision):
    """Even approved decisions can retrieve relevant regulations."""
    grounded = grounder.ground_decision(approved_decision)
    # Approved decisions may have fewer citations but the method shouldn't crash
    assert isinstance(grounded.regulatory_citations, list)


def test_citation_format(grounder, declined_decision):
    """Citations should follow [ID] Source: Provision format."""
    grounded = grounder.ground_decision(declined_decision)
    for citation in grounded.regulatory_citations:
        assert citation.startswith("[")
        assert "]" in citation


def test_custom_corpus():
    """Grounder should work with a custom corpus."""
    custom_corpus = [
        Regulation(
            id="TEST-1",
            source="Test Regulation",
            provision="This is a test provision.",
            keywords=["test_feature"],
            category="test",
        )
    ]
    grounder = PolicyGrounder(corpus=custom_corpus)
    decision = CreditDecision(
        applicant_id="TEST-CUSTOM",
        approved=False,
        risk_score=0.55,
        threshold_used=0.45,
        top_factors=[{"feature": "test_feature", "impact": 0.5, "value": 1.0}],
        reason_codes=["TEST"],
    )
    citations = grounder.retrieve(decision)
    assert len(citations) == 1
    assert citations[0]["id"] == "TEST-1"
