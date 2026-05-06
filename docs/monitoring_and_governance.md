# Monitoring & Data Governance Policy

> **Scope:** This document defines the monitoring, drift detection, and data retention requirements for the AI Credit Decisioning Engine in a production deployment scenario.
>
> **Note:** This is a portfolio project using synthetic data. This policy documents what *would* be required for production deployment, demonstrating operational maturity.

---

## 1. Model Performance Monitoring

### 1.1 Metrics Dashboard (Monthly)

| Metric | Threshold | Action if Breached |
|--------|-----------|-------------------|
| **AUC-ROC** | ≥ 0.78 | Trigger model retrain review |
| **Gini Coefficient** | ≥ 0.56 | Escalate to model risk committee |
| **KS Statistic** | ≥ 0.40 | Investigate feature drift |
| **Brier Score** | ≤ 0.18 | Recalibrate (Platt scaling) |
| **Precision** | ≥ 0.50 | Review threshold setting |
| **Recall** | ≥ 0.60 | Review false negative impact |
| **Approval Rate** | 55%–75% | Business review if outside band |
| **Default Rate (30-day)** | ≤ 18% | Emergency model freeze if exceeded |

### 1.2 Population Stability Index (PSI)

Monitor input feature distributions monthly using PSI:

```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```

| PSI Value | Interpretation | Action |
|-----------|---------------|--------|
| < 0.10 | No significant shift | Continue monitoring |
| 0.10 – 0.25 | Moderate shift | Investigate root cause |
| > 0.25 | Significant shift | Mandatory model retrain |

**Features to monitor:**
- `debt_to_income_ratio`
- `monthly_income`
- `requested_amount`
- `credit_utilization_proxy`
- `late_payment_frequency`
- `employment_tenure_months`

### 1.3 Calibration Monitoring

Compare predicted probabilities to observed default rates in decile buckets:

| Decile | Expected Default Rate | Actual Default Rate | Acceptable Drift |
|--------|----------------------|--------------------|-|
| 1 (lowest risk) | ~2% | ±3pp | Flag if >5% |
| 5 (median) | ~12% | ±5pp | Flag if >17% |
| 10 (highest risk) | ~40% | ±8pp | Flag if >48% |

Recalibrate using `CalibratedClassifierCV` if any decile drifts beyond acceptable range for 2 consecutive months.

---

## 2. Fairness Monitoring

### 2.1 Quarterly Fairness Audit

Re-run the Layer 3 Fairness Auditor on the latest 90 days of decisions:

| Check | Standard | Frequency |
|-------|----------|-----------|
| **4/5ths Rule (AIR)** | AIR ≥ 0.80 across all protected groups | Quarterly |
| **Proxy Variable Detection** | η² < 0.06 for all feature-group pairs | Quarterly |
| **Approval Rate Parity** | Max disparity < 10pp across groups | Monthly |

### 2.2 Escalation Path

```
AIR < 0.80 for any group
    → Immediate escalation to Compliance Officer
    → 30-day remediation deadline
    → Model freeze if not resolved

Proxy variable η² > 0.14 (large effect)
    → Document in model risk register
    → Evaluate feature removal vs. business impact
    → Present to risk committee within 15 days
```

---

## 3. Explanation Quality Monitoring

### 3.1 LLM Grounding Audit (Weekly)

Sample 50 adverse action notices per week and verify:

| Check | Target | Method |
|-------|--------|--------|
| **SHAP Grounding Rate** | ≥ 95% | Do reason codes map to top SHAP features? |
| **Regulatory Citation Accuracy** | 100% | Are cited CBUAE provisions relevant to the decision? |
| **Reason Code Count** | = 4 | ECOA Reg B requires 4 specific reason codes |
| **Hallucination Rate** | 0% | Does the notice contain claims not supported by SHAP or regulations? |

### 3.2 Automated Checks

The `PipelineEvaluator.evaluate()` method runs these checks programmatically. Schedule as a nightly batch job in production.

---

## 4. Data Retention Policy

### 4.1 Retention Schedule

| Data Category | Retention Period | Justification |
|--------------|-----------------|---------------|
| **Credit decisions** (approve/deny + risk score) | 7 years | CBUAE record-keeping requirements |
| **Adverse action notices** | 7 years | Consumer protection compliance |
| **SHAP explanations** (top factors per decision) | 7 years | Auditability — must reproduce decision rationale |
| **Fairness audit reports** | 5 years | Regulatory audit trail |
| **Model artifacts** (joblib + manifest) | Life of model + 3 years | SR 11-7 model risk management |
| **Raw applicant data** | 3 years after last interaction | CBUAE data protection |
| **Training datasets** | Life of model + 1 year | Reproducibility |
| **LLM prompts & responses** | 1 year | Debugging and quality assurance |
| **System logs** (API calls, errors) | 1 year | Operational monitoring |

### 4.2 Data Classification

| Classification | Examples | Handling |
|---------------|----------|----------|
| **PII** | Name, Emirates ID, phone, email | Encrypted at rest (AES-256), masked in logs |
| **Sensitive Financial** | Income, debt, credit score | Encrypted at rest, access-controlled |
| **Protected Attributes** | Age, gender, nationality | Stored only for fairness auditing; never used in scoring |
| **Model Artifacts** | .joblib files, manifests | Version-controlled, immutable after training |
| **Synthetic Data** | Generated applicant records | No retention requirements (not real data) |

### 4.3 Deletion Protocol

1. **Automated expiry:** Cron job checks retention dates monthly
2. **Right to erasure:** Applicant data deletable on request within 30 days (CBUAE consumer protection)
3. **Model retirement:** When a model version is retired, archive artifacts to cold storage for the retention period
4. **Audit log:** All deletions logged with timestamp, data category, and authorizing officer

---

## 5. Incident Response

### 5.1 Model Failure Modes

| Failure Mode | Detection | Response |
|-------------|-----------|----------|
| **Model returns NaN/error** | Real-time health check | Fallback to previous model version |
| **AUC drops below 0.78** | Monthly metrics review | Trigger retrain pipeline |
| **Default rate exceeds 18%** | Weekly portfolio review | Emergency model freeze + manual review |
| **Fairness violation (AIR < 0.80)** | Quarterly audit | Compliance escalation + 30-day fix |
| **LLM hallucination detected** | Weekly grounding audit | Disable LLM, fall back to template notices |
| **Data pipeline failure** | Automated alerting | Halt new decisions until resolved |

### 5.2 Rollback Procedure

```
1. Identify the issue (monitoring alert or manual report)
2. Load previous model version: CreditRiskModel.load_model("data/models/model_v{prev}.joblib")
3. Run validation suite on previous version
4. Switch production pointer to previous version
5. Document incident in model risk register
6. Root cause analysis within 5 business days
```

---

## 6. Regulatory Reporting

| Report | Frequency | Recipient | Content |
|--------|-----------|-----------|---------|
| **Model Performance Report** | Monthly | Risk Committee | AUC, Gini, KS, calibration, approval/default rates |
| **Fairness Audit Report** | Quarterly | Compliance Officer + CBUAE (on request) | AIR by group, proxy variables, remediation actions |
| **Model Change Report** | Per retrain | Model Risk Committee | What changed, why, validation results, before/after metrics |
| **Incident Report** | Per incident | CRO + Compliance | Root cause, impact, remediation, prevention |

---

*This policy should be reviewed and updated annually, or whenever there is a material change to the model, data sources, or regulatory requirements.*
