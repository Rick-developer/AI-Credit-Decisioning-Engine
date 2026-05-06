"""
regulations.py — Curated CBUAE regulatory corpus for policy grounding.

Design decision: We use a curated in-memory knowledge base instead of a full
vector-store RAG pipeline because:
1. The regulatory surface area for BNPL credit decisioning is bounded (~20 provisions)
2. Full RAG (ChromaDB, embeddings, reranking) is already demonstrated in our
   UAE Regulatory Compliance RAG Agent project — repeating it here adds no signal
3. A curated corpus ensures 100% citation accuracy (no retrieval errors)
4. This approach mirrors how production credit systems actually work: compliance
   teams maintain a curated regulation map, not a semantic search over raw PDFs

Each regulation entry includes:
- source: Official regulation name and section number
- provision: The actual regulatory text (paraphrased from public CBUAE guidance)
- keywords: Feature names and concepts that trigger this regulation
- category: Grouping for structured retrieval
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Regulation:
    """A single regulatory provision relevant to credit decisioning."""
    id: str
    source: str
    provision: str
    keywords: List[str] = field(default_factory=list)
    category: str = ""


# ─── CBUAE Regulatory Corpus ───
# Based on publicly available CBUAE Consumer Protection Standards,
# Finance Companies Regulation, and BNPL-specific guidance.

CBUAE_CORPUS: List[Regulation] = [
    # ── Credit Exposure Limits ──
    Regulation(
        id="CBUAE-FC-4.1",
        source="CBUAE Finance Companies Regulation, Article 4.1",
        provision=(
            "The total credit facility extended to any individual borrower by a finance "
            "company shall not exceed AED 20,000 or three times the borrower's verified "
            "monthly net income, whichever is lower."
        ),
        keywords=["requested_amount", "monthly_income", "max_eligible_credit",
                   "requested_to_income_ratio", "credit_limit", "exposure"],
        category="credit_limits",
    ),
    Regulation(
        id="CBUAE-FC-4.2",
        source="CBUAE Finance Companies Regulation, Article 4.2",
        provision=(
            "Finance companies must perform a credit bureau check through Al Etihad "
            "Credit Bureau (AECB) for any borrower whose total credit exposure exceeds "
            "AED 5,000 across all providers."
        ),
        keywords=["requires_bureau_check", "existing_credit_exposure",
                   "credit_utilization_proxy", "bureau"],
        category="credit_limits",
    ),

    # ── Affordability & Responsible Lending ──
    Regulation(
        id="CBUAE-CP-3.1",
        source="CBUAE Consumer Protection Standards, Section 3.1 — Responsible Lending",
        provision=(
            "Licensed financial institutions must assess borrower affordability before "
            "extending credit. The assessment must consider the borrower's income, "
            "existing financial obligations, and living expenses to ensure the borrower "
            "can service the debt without undue financial hardship."
        ),
        keywords=["debt_to_income_ratio", "monthly_obligations", "monthly_income",
                   "post_approval_dti", "spend_to_income_ratio", "affordability"],
        category="affordability",
    ),
    Regulation(
        id="CBUAE-CP-3.2",
        source="CBUAE Consumer Protection Standards, Section 3.2 — Debt Burden Ratio",
        provision=(
            "The total debt burden ratio (DBR) of a borrower — including the proposed "
            "new facility — should not exceed 50% of the borrower's verified monthly "
            "income. Lenders must calculate the post-approval DBR before extending credit."
        ),
        keywords=["debt_to_income_ratio", "post_approval_dti", "monthly_obligations",
                   "monthly_income", "debt_burden"],
        category="affordability",
    ),

    # ── Adverse Action & Transparency ──
    Regulation(
        id="CBUAE-CP-5.1",
        source="CBUAE Consumer Protection Standards, Section 5.1 — Adverse Action Notices",
        provision=(
            "When a credit application is declined, the lender must provide the applicant "
            "with a clear, specific, and written explanation of the reasons for the "
            "decision. Generic or vague reasons (e.g., 'does not meet criteria') are "
            "insufficient. The notice must reference the specific factors that led to "
            "the adverse decision."
        ),
        keywords=["adverse_action_notice", "reason_codes", "declined", "denied",
                   "explanation", "transparency"],
        category="adverse_action",
    ),
    Regulation(
        id="CBUAE-CP-5.2",
        source="CBUAE Consumer Protection Standards, Section 5.2 — Right to Explanation",
        provision=(
            "Borrowers have the right to receive a full explanation of any automated "
            "decision that affects their access to credit. Where algorithmic or AI-based "
            "scoring is used, the lender must be able to explain which factors contributed "
            "to the decision in terms understandable to the borrower."
        ),
        keywords=["explanation", "shap", "top_factors", "algorithmic", "automated",
                   "ai_scoring", "transparency"],
        category="adverse_action",
    ),

    # ── Income & Employment Verification ──
    Regulation(
        id="CBUAE-FC-5.1",
        source="CBUAE Finance Companies Regulation, Article 5.1 — Income Verification",
        provision=(
            "Finance companies must verify the borrower's income through salary "
            "certificates, bank statements, or employer confirmation letters before "
            "extending credit facilities above AED 5,000."
        ),
        keywords=["monthly_income", "employment_type", "employment_tenure_months",
                   "income_verification", "salaried"],
        category="verification",
    ),

    # ── Payment History & Default ──
    Regulation(
        id="CBUAE-CP-6.1",
        source="CBUAE Consumer Protection Standards, Section 6.1 — Payment History Assessment",
        provision=(
            "Lenders must consider the borrower's historical payment behavior, including "
            "any record of late payments, defaults, or delinquencies, as part of the "
            "credit assessment process."
        ),
        keywords=["num_late_payments", "late_payment_frequency", "defaulted",
                   "payment_history", "delinquency"],
        category="credit_history",
    ),
    Regulation(
        id="CBUAE-CP-6.2",
        source="CBUAE Consumer Protection Standards, Section 6.2 — Transaction History",
        provision=(
            "Applicants with limited or no transaction history should be assessed with "
            "additional caution. Lenders may require supplementary documentation or "
            "apply conservative credit limits for thin-file applicants."
        ),
        keywords=["transaction_history_months", "num_previous_applications",
                   "thin_file", "new_to_credit"],
        category="credit_history",
    ),

    # ── Fairness & Non-Discrimination ──
    Regulation(
        id="CBUAE-CP-7.1",
        source="CBUAE Consumer Protection Standards, Section 7.1 — Non-Discrimination",
        provision=(
            "Financial institutions must ensure that credit decisions are made without "
            "discrimination based on gender, nationality, ethnicity, religion, or age. "
            "Lending criteria must be based on objective financial factors."
        ),
        keywords=["gender", "nationality", "age_group", "fairness", "discrimination",
                   "protected_class", "demographic_parity"],
        category="fairness",
    ),
    Regulation(
        id="CBUAE-CP-7.2",
        source="CBUAE Consumer Protection Standards, Section 7.2 — Algorithmic Fairness",
        provision=(
            "Where automated decision-making systems are used for credit assessment, "
            "institutions must periodically audit these systems for unintended bias "
            "against protected demographic groups and document the results."
        ),
        keywords=["fairness", "bias", "proxy_variable", "audit", "demographic_parity",
                   "algorithmic_fairness"],
        category="fairness",
    ),

    # ── BNPL-Specific ──
    Regulation(
        id="CBUAE-BNPL-1.1",
        source="CBUAE BNPL Regulatory Framework, Section 1.1 — Product Classification",
        provision=(
            "Buy-Now-Pay-Later products are classified as consumer credit facilities "
            "under the Finance Companies Regulation. All BNPL providers must hold a "
            "valid CBUAE license and comply with all applicable consumer credit regulations."
        ),
        keywords=["bnpl", "requested_amount", "credit_facility", "license"],
        category="bnpl",
    ),
    Regulation(
        id="CBUAE-BNPL-2.1",
        source="CBUAE BNPL Regulatory Framework, Section 2.1 — Affordability Assessment",
        provision=(
            "BNPL providers must perform a proportionate affordability assessment for "
            "every transaction. For facilities below AED 1,000, a simplified assessment "
            "based on available data is acceptable. For facilities above AED 1,000, a "
            "full affordability assessment including income verification is required."
        ),
        keywords=["requested_amount", "affordability", "monthly_income",
                   "spend_to_income_ratio", "assessment"],
        category="bnpl",
    ),

    # ── Model Risk Management (SR 11-7 aligned) ──
    Regulation(
        id="SR-11-7-MRM",
        source="Federal Reserve SR 11-7 — Guidance on Model Risk Management (Reference)",
        provision=(
            "Financial institutions using models for credit decisioning must maintain "
            "comprehensive model documentation (model cards), perform independent "
            "validation, and conduct ongoing performance monitoring. Models must be "
            "tested for accuracy, stability, and potential for discriminatory outcomes."
        ),
        keywords=["model_card", "validation", "monitoring", "documentation",
                   "model_risk", "calibration"],
        category="model_governance",
    ),
]
