# PRD v1.0: AI Credit Decisioning Engine

**Author:** Mousumee  
**Date:** 2026-05-04  
**Status:** Draft v1.0  
**Version:** 1.0  

---

> [!IMPORTANT]
> ## Scope Declaration
> **This is a portfolio demonstration system**, not a production credit decisioning platform.
> - All applicant data is **synthetically generated**. No real consumer credit data is used.
> - The system produces **advisory risk assessments only**. It cannot and should not replace licensed credit bureaus, underwriting teams, or regulatory review.
> - CBUAE regulatory references are based on publicly available regulations and are used to demonstrate RAG-grounded explainability, not to provide legal or compliance advice.
> - The target audience is hiring managers evaluating AI product design, ML/LLM architecture decisions, and fairness-aware system thinking.

---

## 1. Executive Summary

UAE's BNPL/lending market — led by Tabby (valued at $1.5B+) and Tamara ($1B+) — is one of the fastest-growing consumer credit segments in the GCC. CBUAE's Finance Companies Regulation now mandates that BNPL providers:
- Cap total credit at **AED 20,000** or 3 months' net income (whichever is lower)
- Perform **credit bureau checks** for any borrower with >AED 5,000 total exposure  
- Provide **specific, actionable reasons** for adverse credit decisions

Standard ML-based credit scoring (XGBoost + feature engineering) solves the *scoring* problem. But the **hard problem** in credit decisioning isn't scoring — it's **explaining**. ML can output "probability of default = 0.73" but cannot write: *"Your application was declined because your monthly obligations exceed 40% of verified income, which falls outside our lending criteria under CBUAE Consumer Protection Standards."*

The **AI Credit Decisioning Engine** is a hybrid ML + LLM pipeline that:
1. **Scores risk** using a calibrated XGBoost model (Layer 1 — commodity baseline)
2. **Generates compliance-grade explanations** by translating SHAP feature attributions into natural language adverse action notices (Layer 2 — the differentiator)
3. **Audits fairness** by detecting bias across protected classes and generating narrative audit reports (Layer 3)
4. **Grounds every claim** in CBUAE regulations via RAG retrieval (Layer 4)

**This project demonstrates:** ML model design, threshold optimization, LLM-powered explainability, fairness/bias analysis, RAG-grounded compliance, and the product decisions involved in building an ethical AI system for a regulated domain.

---

## 2. Problem Statement

### Context

UAE's consumer credit landscape has evolved rapidly:

- **50+ BNPL/lending fintechs** operate in the UAE/GCC, all requiring real-time credit decisioning
- **Tabby** built an "AI Factory" with NVIDIA HGX infrastructure specifically for risk scoring and fraud detection
- **CBUAE** introduced comprehensive BNPL regulation under the Finance Companies framework, with strict limits and transparency requirements
- Many UAE consumers have **thin credit files** (no traditional credit history), making alternative data and ML-based scoring critical

### The Gap

| Problem | Impact |
|---------|--------|
| ML scoring is commoditized | Every fintech uses XGBoost/LightGBM — it doesn't differentiate |
| SHAP values are for engineers, not regulators | A compliance officer can't action "feature_3 contributed -0.42" |
| Adverse action notices require specific reasons | Generic denials violate consumer protection standards |
| Fairness testing is done ad-hoc, if at all | No structured bias detection or audit trail |
| Compliance claims aren't grounded in regulations | LLMs can hallucinate regulatory references |

### Why Current Solutions Fail

- **Pure ML scoring:** Produces probabilities but no human-readable explanations
- **Rule-based explanations:** Rigid, can't adapt to model changes, miss nuanced feature interactions
- **Free-form LLM explanations:** Risk hallucinating compliance claims that don't map to actual regulations
- **Manual review:** Doesn't scale to instant BNPL decisioning at checkout

---

## 3. User Personas

### Primary: Credit Operations Lead at a UAE BNPL Startup

**Nadia, 34** — Head of Credit Risk at a Series B BNPL company in Dubai. She manages a team of 3 risk analysts and reports to the CFO.

- **Needs:** Instant credit decisions with explainable reasons for every approval and denial
- **Pain:** Current ML model gives her probabilities but she writes adverse action notices manually
- **Constraint:** CBUAE auditors expect specific, traceable reasons — not "the model said no"
- **Quota:** Must maintain approval rate >65% while keeping default rate <18%

### Secondary: CBUAE Compliance Auditor

**Khalid, 42** — Senior Examiner at CBUAE responsible for reviewing BNPL provider compliance.

- **Needs:** Evidence that credit decisions are explainable, fair, and grounded in regulatory requirements
- **Pain:** Fintechs show him ML model performance charts but can't explain individual decisions
- **Concern:** Algorithmic bias across nationality or gender

---

## 4. Feature Scope (MoSCoW)

### Must Have (MVP)

| Feature | Layer | Description |
|---------|-------|-------------|
| Synthetic data generator | 0 | 5,000 realistic UAE BNPL applicants with protected classes |
| Feature engineering pipeline | 1 | Income ratios, obligation burden, employment stability, spend velocity |
| XGBoost risk scorer | 1 | Calibrated probabilities with SHAP attributions |
| Threshold optimization | 1 | Interactive approval rate vs default rate trade-off |
| LLM adverse action notices | 2 | SHAP → structured adverse action explanation in natural language |
| Structured explanation output | 2 | JSON with reason codes, explanation text, contributing factors |
| Demographic parity testing | 3 | Approval rate comparison across age, gender, nationality |
| Streamlit dashboard | UI | Submit applicant → score + explanation + fairness metrics |
| Unit tests | QA | pytest coverage for all core modules |

### Should Have

| Feature | Layer | Description |
|---------|-------|-------------|
| RAG policy grounding | 4 | Ground LLM explanations in CBUAE regulatory text |
| Equalized odds testing | 3 | Equal FPR/FNR across protected groups |
| Proxy variable detection | 3 | Identify features that correlate with protected classes |
| LLM fairness narratives | 3 | Generate natural language audit reports from statistical tests |
| Calibration plot | 1 | Visual proof that predicted probabilities are well-calibrated |

### Could Have

| Feature | Layer | Description |
|---------|-------|-------------|
| Model comparison (LR vs XGBoost) | 1 | Side-by-side interpretability vs accuracy trade-off |
| A/B threshold simulator | UI | Compare business impact of different approval thresholds |
| Batch scoring mode | 1 | Upload CSV → batch risk assessment with explanations |

### Won't Have (Not in Scope)

| Feature | Reason |
|---------|--------|
| Real credit bureau integration | No access to AECB data; synthetic data only |
| Real-time API endpoint | Portfolio demo, not production deployment |
| Multi-model ensemble | Adds complexity without PM learning signal |
| Arabic language explanations | English-only for portfolio demonstration |

---

## 5. Architecture

### 4-Layer Hybrid Design

```
Layer 1: ML Risk Scoring (Commodity Baseline)
┌──────────────────────────────────────────┐
│  Applicant Data → Feature Engineering    │
│  → XGBoost Classifier → P(default)      │
│  → SHAP Feature Attributions             │
└────────────┬─────────────────────────────┘
             │ SHAP values + prediction
             ▼
Layer 2: LLM Explanation Engine (Differentiator)
┌──────────────────────────────────────────┐
│  SHAP values → Structured prompt         │
│  → Groq/Llama3-70B → Adverse action     │
│    notice in natural language             │
│  → The LLM NEVER decides — only explains │
└────────────┬─────────────────────────────┘
             │ explanation text
             ▼
Layer 3: Fairness Audit (PM Signal)
┌──────────────────────────────────────────┐
│  Model predictions × protected classes   │
│  → Statistical parity tests              │
│  → Proxy variable detection              │
│  → LLM-generated audit narratives        │
└────────────┬─────────────────────────────┘
             │ fairness report
             ▼
Layer 4: RAG Policy Grounding (Trust Layer)
┌──────────────────────────────────────────┐
│  Explanation text → RAG retrieval        │
│  → Match claims to CBUAE regulations     │
│  → Citation verification                 │
│  → Grounded explanation with references  │
└──────────────────────────────────────────┘
```

### Key Architectural Decision: The LLM Never Decides

The LLM's role is **strictly translational**. It converts SHAP feature attributions (which are deterministic and auditable) into human-readable compliance language. The credit decision itself always comes from the ML model. This means:

- **Auditability:** Every explanation traces back to specific SHAP values
- **Consistency:** Same inputs → same ML decision (the LLM adds language variation, not decision variation)
- **Regulatory safety:** We can show auditors exactly which features drove the decision and by how much

---

## 6. Success Metrics

### Business Metrics (What a PM Would Track)

| Metric | Target | Why It Matters |
|--------|--------|---------------|
| Approval rate at default_rate < 18% | ≥ 65% | Business viability — too few approvals = no revenue |
| Explanation grounding rate | ≥ 90% | % of LLM-generated claims traceable to SHAP values or regulations |
| Fairness: max approval rate disparity | < 10pp | Between any two demographic groups — regulatory threshold |
| Adverse action notice specificity | 3+ specific reasons | Each denial must cite 3+ concrete factors (not generic text) |

### Technical Metrics (Supporting Evidence)

| Metric | Target | Why It Matters |
|--------|--------|---------------|
| AUC-ROC | ≥ 0.80 | Model discriminative ability |
| Calibration (Brier score) | ≤ 0.15 | Predicted probabilities match actual default rates |
| SHAP fidelity | 100% | Every explanation factor maps to a real SHAP value |
| Test coverage | ≥ 85% | Code quality signal |

---

## 7. Decisions Not Taken

| Decision | What We Chose | What We Rejected | Why |
|----------|--------------|------------------|-----|
| LLM for explanation, not scoring | LLM translates SHAP → language | LLM as credit decision-maker | LLMs are non-deterministic; credit decisions must be reproducible and auditable |
| XGBoost over deep learning | XGBoost + SHAP | Neural network scorer | SHAP works natively with tree-based models; neural network SHAP is approximate and slow |
| Synthetic data over real data | Programmatic generation | Kaggle credit datasets | Full control over protected classes, UAE-specific features, CBUAE-compliant distributions |
| Groq over OpenAI | Groq + Llama3-70B (free tier) | GPT-4 ($30/1M tokens) | $0 cost, sufficient quality for structured explanation generation, consistent with portfolio |
| Demographic parity as primary fairness metric | Equal approval rates across groups | Equalized odds | More intuitive for non-technical stakeholders; equalized odds as secondary |
| Substring citation verification | Match explanation claims to SHAP | Trust LLM claims blindly | Same safety pattern proven in UAE RAG Agent — never trust LLM-generated compliance claims |

---

## 8. Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|------------|
| Synthetic data too simple | Model overfits, unrealistic performance | Multi-modal distributions, realistic noise, correlation structure |
| LLM hallucinates compliance claims | Portfolio credibility damage | RAG grounding + citation verification (Layer 4) |
| Fairness metrics look good by accident | False sense of compliance | Test across multiple fairness definitions, document limitations |
| XGBoost AUC too high on synthetic data | Looks "too good" to hiring committees | Document clearly that synthetic data enables high scores; production would require recalibration |

---

## 9. Technical Constraints

- **Python 3.10+** with type hints
- **No real consumer data** — fully synthetic
- **$0 inference cost** — Groq free tier for LLM
- **Local-first** — all ML models run locally
- **Streamlit** for demo UI
- **pytest** for testing

---

## 10. Build Timeline

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 1. Scaffolding + PRD | Project structure, PRD.md | ✅ Complete |
| 2. Synthetic Data | 5,000 UAE BNPL applicant records | ✅ Complete |
| 3. Feature Engineering | Engineered risk signals | ✅ Complete |
| 4. ML Scoring | XGBoost + SHAP + threshold optimization | ✅ Complete |
| 5. LLM Explanations | Adverse action notice generator | ✅ Complete |
| 6. Fairness Audit | Bias detection + audit narratives | ✅ Complete |
| 7. Streamlit Dashboard | Interactive demo UI | ✅ Complete |
| 8. Evaluation | Triple evaluation framework | ✅ Complete |
| 9. Documentation | README + case study + Notion 10/10 | ✅ Complete |
| 10. Audit Remediation | Security hardening, ML calibration, Layer 4, model card | ✅ Complete |

---

*This PRD follows the standards established in `CLAUDE.md` and builds on patterns proven in the UAE Regulatory Compliance RAG Agent (Project 2).*
