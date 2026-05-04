# ⚖️ AI Credit Decisioning Engine (Hybrid ML+LLM Architecture)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-orange.svg)](https://shap.readthedocs.io/)
[![Groq Llama3](https://img.shields.io/badge/Groq-Llama3_8B-black.svg)](https://groq.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)

A production-ready AI Credit Engine built for the UAE BNPL (Buy Now, Pay Later) market. This repository demonstrates a highly scalable **4-Layer Hybrid AI Architecture** that solves the industry's hardest problem: deploying AI in highly regulated financial environments without sacrificing compliance or fairness.

---

## 🎯 The Product Vision

Standard ML models optimize strictly for accuracy (AUC). However, in credit decisioning, **accuracy is not enough**. A model must be explainable (to generate regulatory Adverse Action Notices) and fair (to prevent discriminatory lending).

Instead of relying on a single "black box" model or an unpredictable LLM, I designed a **deterministic, hybrid architecture**.

### 🏗️ The 4-Layer Architecture

1. **Layer 1: Deterministic Risk Scoring (XGBoost).** High-accuracy tabular prediction. 
2. **Layer 2: Explainability (SHAP).** Extracts the exact mathematical drivers of every decision.
3. **Layer 3: Generative Compliance (Llama 3 via Groq).** Translates the raw SHAP mathematics into human-readable Adverse Action Notices. **Crucial PM Decision:** The LLM *never* makes the credit decision; it only explains it.
4. **Layer 4: Fairness Auditor.** Scans for Demographic Parity and Proxy Variable Bias across protected classes (Age, Gender, Nationality).

```mermaid
graph TD
    A[Raw Applicant Data] --> B[Feature Engineering Pipeline]
    B --> C{CBUAE Compliance Check}
    C -- Fails Limits --> D[Auto-Decline]
    C -- Passes --> E[XGBoost Risk Model]
    
    E --> F[SHAP Explainer]
    F --> G[LLM Translation Layer Groq/Llama3]
    G --> H[Consumer Adverse Action Notice]
    
    E -.-> I[Fairness Auditor Layer]
    I -.-> J[Demographic Parity Report]
    I -.-> K[Proxy Variable Detection]
    
    classDef model fill:#f9f,stroke:#333,stroke-width:2px;
    classDef llm fill:#bbf,stroke:#333,stroke-width:2px;
    classDef compliance fill:#bfb,stroke:#333,stroke-width:2px;
    
    E class:model
    F class:model
    G class:llm
    I class:compliance
    C class:compliance
```

---

## 🔑 Key Product Management Decisions

### 1. "Fairness Through Unawareness" is a Flawed Strategy
**The Problem:** The industry standard for avoiding bias is to simply drop protected classes (e.g., `gender`, `nationality`) from the training dataset.
**The Insight:** Machine learning models are exceptionally good at finding proxy variables. For instance, in our UAE synthetic dataset, the model learned to discriminate based on `age_group` by using `employment_tenure_months` as a proxy.
**The Solution:** I built an automated Fairness Auditor (Layer 4) that uses ANOVA to detect these highly correlated proxy variables, proving that explicitly measuring bias is safer than ignoring it.

### 2. Bounding LLM Hallucinations in FinServ
**The Problem:** You cannot let an LLM decide who gets a loan. It is non-deterministic and impossible to audit.
**The Solution:** The LLM is restricted to a pure translation role. It is fed only the top 3 risk-increasing variables mathematically extracted by SHAP (Layer 2). The LLM's prompt instructs it to translate these variables into empathetic, consumer-friendly language. If the LLM fails, the system falls back to a deterministic reason code.

### 3. CBUAE Regulatory Hard-Coding
**The Problem:** AI models shouldn't learn regulations from data; they should be bounded by them.
**The Solution:** The feature engineering pipeline hard-codes UAE Central Bank constraints (e.g., maximum AED 20,000 BNPL cap, mandatory Al Etihad Credit Bureau checks for exposures > AED 5,000) before the data ever reaches the XGBoost model.

---

## 🛠️ Technical Implementation

### Prerequisites
- Python 3.10+
- `GROQ_API_KEY` (for the LLM Explanation Layer)

### Installation
```bash
git clone https://github.com/yourusername/AI-Credit-Decisioning-Engine.git
cd AI-Credit-Decisioning-Engine
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### 1. Generate UAE-Specific Synthetic Data
Generates 5,000 highly correlated synthetic records tailored to the UAE demographic distribution (88% expat, 12% Emirati).
```bash
python src/data/generator.py
```

### 2. Run the Test Suite
The repository includes a comprehensive testing framework covering feature engineering, risk scoring, LLM integration, and fairness.
```bash
pytest tests/ -v
```

### 3. Launch the Streamlit Dashboard
Interact with the full 4-layer pipeline via a clean, unified dashboard.
```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env
streamlit run app.py
```

---

## 📊 Triple-Evaluation Framework

A modern AI product must be evaluated across multiple dimensions. This engine implements:

| Evaluation Dimension | Metric Used | Purpose |
| :--- | :--- | :--- |
| **1. ML Accuracy** | AUC-ROC, Brier Score | Ensures the model actually predicts default risk accurately. |
| **2. Explanation Grounding** | SHAP Fidelity Rate | Ensures the LLM's natural language notice maps 100% to the mathematical drivers. |
| **3. Portfolio Fairness** | Max Disparity < 15% | Ensures approval rates don't illegally skew against protected classes. |

---

## 🤝 Let's Connect
I'm a Data/Technical Product Manager specializing in AI architecture, data platforms, and MLOps. If you're building products at the intersection of complex data and rigorous compliance, let's talk.

[LinkedIn](https://linkedin.com/in/yourprofile) | [Portfolio](https://your-notion-portfolio.com)
