# Model Card: UAE BNPL Credit Risk Scorer

> Prepared in alignment with [SR 11-7: Guidance on Model Risk Management](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm) and the [CBUAE Consumer Protection Regulation](https://www.centralbank.ae).

---

## Model Overview

| Field | Value |
|-------|-------|
| **Model Name** | BNPL Credit Risk Scorer v1.0 |
| **Model Type** | Supervised binary classification |
| **Algorithm** | XGBoost (gradient-boosted decision trees) + Platt scaling calibration |
| **Framework** | `xgboost==1.7+`, `scikit-learn`, `shap` |
| **Task** | Predict probability of default (`P(default)`) for UAE BNPL applicants |
| **Output** | Calibrated probability ∈ [0.0, 1.0] |
| **Decision Threshold** | 0.45 (configurable; applicant approved if `P(default) ≤ threshold`) |
| **Date** | May 2026 |
| **Owner** | Gyana Pattnaik |

---

## Intended Use

### Primary Use Case
Automated first-pass credit decisioning for Buy-Now-Pay-Later (BNPL) applications in the UAE market, operating under CBUAE regulatory constraints:
- **Maximum credit exposure:** AED 20,000 or 3× monthly income (whichever is lower)
- **Bureau check trigger:** Total exposure > AED 5,000

### Intended Users
1. **Credit risk analysts** — reviewing model decisions and SHAP explanations
2. **Compliance officers** — auditing fairness reports and adverse action notices
3. **Product managers** — monitoring approval rates and portfolio risk

### Out-of-Scope Uses
- ❌ Fully autonomous credit decisions without human review
- ❌ Markets outside UAE (data distributions differ)
- ❌ Credit products beyond BNPL (mortgages, auto loans, credit cards)
- ❌ Use as the sole decision-maker without the explanation layer (Layer 2)

---

## Training Data

### Dataset
| Property | Value |
|----------|-------|
| **Source** | Synthetic data generator (`src/data/generator.py`) |
| **Size** | 5,000 applicant records |
| **Split** | 80% train / 20% test (stratified by target) |
| **Target Variable** | `defaulted` (binary: 0 = no default, 1 = defaulted) |
| **Default Rate** | ~15% (class imbalance handled via `scale_pos_weight`) |

### Why Synthetic Data?
This is a portfolio demonstration project. Real credit data is:
1. Regulated under CBUAE data protection requirements
2. Classified as personal financial information under UAE Federal Law
3. Not available for open-source projects

The synthetic generator models realistic UAE BNPL demographics, income distributions, and default correlations to produce statistically representative data.

### Demographic Distribution
| Protected Attribute | Groups | Distribution |
|----|---|---|
| **Gender** | Male, Female | 65% / 35% (UAE workforce skew) |
| **Nationality** | Emirati, South Asian, SE Asian, Arab Expat, Western, Other | 12% / 45% / 15% / 15% / 8% / 5% |
| **Age Group** | 18-25, 26-35, 36-45, 46-60 | Uniform ages 18-65, bucketed |

### Known Biases in Training Data
The data generator **deliberately injects correlations** between protected attributes and financial features to simulate real-world disparities:
- **Nationality → Income:** Western nationals receive a 1.5× income multiplier; South Asian nationals 0.8×
- **Age → Employment Tenure:** Strong positive correlation (proxy variable)

These biases are intentional — they exist so Layer 3 (Fairness Auditor) can detect them.

---

## Features

### Raw Input Features (14)
| Feature | Type | Description |
|---------|------|-------------|
| `age` | int | Applicant age (18-65) |
| `employment_type` | categorical | Salaried / Self-employed / Freelancer / Unemployed |
| `employment_tenure_months` | int | Months at current employer |
| `monthly_income` | float | Net monthly income (AED) |
| `monthly_obligations` | float | Existing monthly debt payments (AED) |
| `monthly_spend` | float | Average monthly spending (AED) |
| `existing_credit_exposure` | float | Total existing credit (AED) |
| `transaction_history_months` | int | Months of account history |
| `num_previous_applications` | int | Prior BNPL/credit applications |
| `num_late_payments` | int | Historical late payment count |
| `requested_amount` | float | BNPL amount requested (AED) |

### Engineered Features (6)
| Feature | Formula | Rationale |
|---------|---------|-----------|
| `debt_to_income_ratio` | `monthly_obligations / monthly_income` | CBUAE responsible lending signal |
| `spend_to_income_ratio` | `monthly_spend / monthly_income` | Free cash flow indicator |
| `requested_to_income_ratio` | `requested_amount / monthly_income` | Affordability check |
| `post_approval_dti` | `(obligations + request/4) / income` | Projected DTI if approved |
| `credit_utilization_proxy` | `existing_exposure / max_eligible_credit` | Credit usage intensity |
| `late_payment_frequency` | `late_payments / max(1, history_months)` | Normalized delinquency rate |

### Excluded Features (Fairness Through Unawareness)
| Feature | Reason for Exclusion |
|---------|---------------------|
| `age_group` | Protected class — age discrimination |
| `gender` | Protected class — gender discrimination |
| `nationality` | Protected class — national origin discrimination |
| `max_eligible_credit` | Target leakage — derived from `monthly_income` (C-3 audit fix) |
| `applicant_id` | Identifier, not predictive |
| `defaulted` | Target variable |
| `requires_bureau_check` | Derived from target-adjacent fields |

---

## Model Architecture

### XGBoost Configuration
```python
XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    max_depth=4,
    learning_rate=0.05,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=<auto: neg_count/pos_count>,
    random_state=42
)
```

### Calibration
Post-training Platt scaling via `CalibratedClassifierCV(cv=5, method='sigmoid')` ensures predicted probabilities are well-calibrated (measured by Brier score).

### Explainability
SHAP `TreeExplainer` computes feature attributions for every prediction. The top 5 factors (by absolute SHAP value) are surfaced in the `CreditDecision` output.

---

## Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **AUC-ROC** | ~0.85+ | Strong discrimination between defaulters and non-defaulters |
| **Gini Coefficient** | ~0.70+ | 2×AUC - 1; industry standard for credit scoring |
| **KS Statistic** | ~0.55+ | Maximum separation between cumulative distributions |
| **Brier Score** | < 0.15 | Well-calibrated probabilities (lower = better) |
| **Precision** | ~0.60+ | Of predicted defaults, percentage actually defaulting |
| **Recall** | ~0.70+ | Of actual defaults, percentage correctly identified |

> **Note:** Exact values vary with each training run due to random splits. Metrics are computed on the held-out 20% test set.

---

## Fairness Assessment

### Methodology
The Fairness Auditor (Layer 3) evaluates demographic parity using the **4/5ths rule** (ECOA / EEOC industry standard):

```
Adverse Impact Ratio = min_group_approval_rate / max_group_approval_rate
Pass condition: AIR ≥ 0.80
```

### Proxy Variable Detection
Even though protected classes are excluded from training features, the auditor performs ANOVA-based proxy detection with **eta-squared effect size** (η² > 0.06 = medium effect) to identify features that encode demographic information indirectly.

### Known Fairness Risks
| Risk | Status | Mitigation |
|------|--------|------------|
| `employment_tenure_months` correlates with `age_group` | ⚠️ Known proxy | Detected by Layer 3; documented for risk committee review |
| `monthly_income` varies by `nationality` | ⚠️ Known proxy | Detected by Layer 3; demonstrates "Fairness Through Unawareness" is insufficient |
| Age group disparity in approval rates | ⚠️ Expected | Young applicants (18-25) have fewer financial signals → higher rejection rates |

### Design Decision
The model deliberately uses "Fairness Through Unawareness" (dropping protected classes) to demonstrate that **this approach alone is insufficient**. The Fairness Auditor exists specifically to prove this point and flag proxy bias, creating a discussion artifact for risk committees.

---

## Ethical Considerations

1. **Synthetic data only** — No real consumer data is used or stored
2. **Not a production system** — This is a portfolio project demonstrating architectural patterns
3. **Human-in-the-loop required** — The system is designed with the expectation of human review before any credit decision is finalized
4. **Adverse Action Notices** — All denials include 4 specific reason codes per ECOA Regulation B requirements
5. **LLM never decides** — The LLM (Layer 2) only translates SHAP values into natural language; it has zero influence on the approve/deny decision

---

## Limitations

| Limitation | Impact | Mitigation Path |
|------------|--------|-----------------|
| Synthetic training data | Model cannot generalize to real applicant distributions | Replace with anonymized real data in production |
| No temporal validation | Model may not capture time-dependent risk patterns | Implement walk-forward validation |
| Single-point calibration | Calibration may degrade as population shifts | Add monitoring + periodic recalibration |
| No model versioning | Cannot A/B test or rollback model versions | Implement MLflow or similar (future work) |
| Layer 4 (RAG Policy Grounding) not implemented | `regulatory_citations` field is always empty | Planned for next iteration |
| UAE-specific only | Income distributions, regulations, and demographics are UAE-specific | Would require full retrain for other markets |

---

## Monitoring Recommendations (Production Deployment)

If this model were deployed to production, the following monitoring would be required:

1. **Population Stability Index (PSI)** — Detect input distribution drift monthly
2. **Fairness metrics recalculation** — Re-run 4/5ths rule quarterly
3. **Calibration check** — Compare predicted vs. actual default rates monthly
4. **Feature drift detection** — Monitor each feature's distribution for shift
5. **Reason code audit** — Sample 50 adverse action notices per week for LLM grounding accuracy

---

## Regulatory Alignment

| Regulation | Requirement | Status |
|------------|-------------|--------|
| **CBUAE Consumer Protection** | Credit cap AED 20K / 3× income | ✅ Enforced in data models |
| **CBUAE Bureau Check** | Required above AED 5K exposure | ✅ `requires_bureau_check` field |
| **ECOA Regulation B** | 4 specific reason codes per denial | ✅ Implemented in explanation engine |
| **SR 11-7 Model Risk Management** | Model card documentation | ✅ This document |
| **EU AI Act (reference)** | High-risk AI system documentation | ✅ Architecture + fairness audit |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-05-04 | Initial model: XGBoost + SHAP + LLM explanations |
| 1.1 | 2026-05-05 | Audit remediation: Platt calibration, 4/5ths rule, target leakage fix, class imbalance handling, KS/Gini metrics |
