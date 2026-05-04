import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from dotenv import load_dotenv

from src.models.scorer import CreditRiskModel
from src.explanations.engine import ExplanationEngine
from src.fairness.auditor import FairnessAuditor
from src.data.models import Applicant

# Set page config
st.set_page_config(
    page_title="UAE AI Credit Engine",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the dashboard
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .adverse-notice {
        background-color: #ffebee;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #d32f2f;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .approved-notice {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2e7d32;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .reason-code {
        background-color: #e0e0e0;
        padding: 4px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9em;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Data & Models ───
@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/sample/applicants.csv")
    except FileNotFoundError:
        st.error("Dataset not found. Run `python src/data/generator.py` first.")
        st.stop()

@st.cache_resource
def load_model(_df):
    model = CreditRiskModel()
    model.train(_df)
    return model

@st.cache_resource
def load_fairness_auditor():
    return FairnessAuditor()

df = load_data()
model = load_model(df)
fairness_auditor = load_fairness_auditor()

# ─── Sidebar Controls ───
st.sidebar.title("⚙️ Engine Configuration")

# Threshold slider
threshold = st.sidebar.slider(
    "Approval Risk Threshold", 
    min_value=0.1, max_value=0.9, value=0.45, step=0.05,
    help="Maximum risk score allowed for approval"
)

# LLM Mock Mode
load_dotenv()
api_key_available = bool(os.environ.get("GROQ_API_KEY"))
use_mock_llm = st.sidebar.checkbox("Use Mock LLM (No API calls)", value=not api_key_available)

if not use_mock_llm and not api_key_available:
    st.sidebar.warning("GROQ_API_KEY not found in .env. Falling back to mock mode.")
    use_mock_llm = True

explanation_engine = ExplanationEngine(api_key=None if use_mock_llm else os.environ.get("GROQ_API_KEY"))

# Select Applicant
st.sidebar.markdown("---")
st.sidebar.subheader("👤 Applicant Selection")

applicant_ids = df['applicant_id'].tolist()
selected_id = st.sidebar.selectbox("Select Applicant ID", applicant_ids, index=0)

applicant_data = df[df['applicant_id'] == selected_id].iloc[0].to_dict()
applicant_df = df[df['applicant_id'] == selected_id]

# Validate against Pydantic model
try:
    applicant_obj = Applicant(**applicant_data)
except Exception as e:
    st.error(f"Data validation error for {selected_id}: {e}")
    st.stop()

# ─── Main Content ───
st.title("⚖️ AI Credit Decisioning Engine")
st.markdown("""
*Hybrid ML+LLM pipeline demonstrating scoring, explainability, and fairness auditing for UAE BNPL applications.*
""")

tab1, tab2, tab3 = st.tabs(["📊 Individual Decisioning", "🔎 LLM Explanation Layer", "⚖️ Fairness Audit (Layer 3)"])

with tab1:
    st.subheader(f"Profile: {selected_id}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Requested Amount", f"AED {applicant_obj.requested_amount:,.0f}")
    col2.metric("Monthly Income", f"AED {applicant_obj.monthly_income:,.0f}")
    col3.metric("Existing Debt", f"AED {applicant_obj.monthly_obligations:,.0f}")
    
    # CBUAE Constraint check
    max_credit = applicant_obj.max_eligible_credit
    utilization = (applicant_obj.existing_credit_exposure + applicant_obj.requested_amount) / max_credit if max_credit > 0 else 1.0
    col4.metric("CBUAE Exposure Limit", f"{utilization:.0%}")
    
    st.markdown("---")
    
    # Run ML Prediction
    with st.spinner("Running ML risk scoring..."):
        decision = model.predict_with_explanation(applicant_df, threshold=threshold)
    
    col_score, col_result = st.columns(2)
    
    with col_score:
        st.subheader("Layer 1: ML Risk Score")
        
        # Gauge chart for risk score
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = decision.risk_score,
            title = {'text': "Probability of Default"},
            gauge = {
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkgray"},
                'steps': [
                    {'range': [0, threshold], 'color': "lightgreen"},
                    {'range': [threshold, 1], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold
                }
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
    with col_result:
        st.subheader("Decision")
        if decision.approved:
            st.success("✅ **APPROVED**")
            st.markdown(f"Risk score ({decision.risk_score:.2f}) is below the threshold ({threshold:.2f}).")
            if applicant_obj.requires_bureau_check:
                st.warning("⚠️ **CBUAE Mandate:** Bureau check required (Total exposure > AED 5,000)")
        else:
            st.error("❌ **DECLINED**")
            st.markdown(f"Risk score ({decision.risk_score:.2f}) exceeds the threshold ({threshold:.2f}).")

    st.subheader("Feature Importance (SHAP)")
    st.markdown("Top factors driving this specific decision:")
    
    if decision.top_factors:
        factors_df = pd.DataFrame(decision.top_factors)
        
        # Create waterfall-like bar chart
        factors_df['color'] = np.where(factors_df['impact'] > 0, 'Risk Increasing', 'Risk Decreasing')
        fig2 = px.bar(
            factors_df, 
            x='impact', 
            y='feature', 
            orientation='h',
            color='color',
            color_discrete_map={'Risk Increasing': '#EF553B', 'Risk Decreasing': '#00CC96'},
            text_auto='.3f'
        )
        fig2.update_layout(height=300, yaxis={'categoryorder':'total ascending'}, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No SHAP factors available.")

with tab2:
    st.subheader("Layer 2: LLM Explanation Engine")
    st.markdown("""
    *The LLM translates the raw SHAP feature attributions into a human-readable adverse action notice. 
    **The LLM never makes the credit decision.** It only explains the ML model's decision.*
    """)
    
    # Generate Explanation
    with st.spinner("Translating SHAP values via LLM..."):
        explained_decision = explanation_engine.generate_explanation(applicant_obj, decision)
    
    if explained_decision.approved:
        st.markdown(f'<div class="approved-notice"><h4>{explained_decision.adverse_action_notice}</h4></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="adverse-notice"><h4>{explained_decision.adverse_action_notice}</h4></div>', unsafe_allow_html=True)
        
    st.markdown("**Structured Reason Codes:**")
    codes_html = " ".join([f'<span class="reason-code">{c}</span>' for c in explained_decision.reason_codes])
    st.markdown(codes_html, unsafe_allow_html=True)
    
    with st.expander("View LLM Prompt Context"):
        st.code(explanation_engine._build_prompt(applicant_obj, decision))

with tab3:
    st.subheader("Layer 3: Fairness Auditor")
    st.markdown("""
    *We explicitly dropped `age_group`, `gender`, and `nationality` during ML training to demonstrate **'Fairness Through Unawareness'**. 
    This audit proves whether the model still learned to discriminate via proxy variables.*
    """)
    
    if st.button("Run Global Fairness Audit"):
        with st.spinner("Scoring entire portfolio and computing demographic parity..."):
            # Score everyone
            df_scored = model.batch_predict(df)
            
            # Check parity
            reports = fairness_auditor.check_demographic_parity(df_scored, threshold=threshold)
            
            # Detect proxies
            proxies = fairness_auditor.detect_proxy_variables(df)
            
            st.markdown("### 1. Demographic Parity (Approval Rates)")
            for r in reports:
                if r.passes_threshold:
                    st.success(r.narrative)
                else:
                    st.error(r.narrative)
                    
                # Chart
                rates = pd.Series(r.group_results).reset_index()
                rates.columns = [r.protected_attribute, 'Approval Rate']
                fig3 = px.bar(rates, x=r.protected_attribute, y='Approval Rate', color=r.protected_attribute)
                fig3.update_layout(height=300, showlegend=False)
                fig3.update_yaxes(tickformat=".1%")
                st.plotly_chart(fig3, use_container_width=True)
                
            st.markdown("### 2. Proxy Variable Detection")
            st.markdown("Features that the model uses which are highly correlated with protected classes:")
            
            if proxies:
                st.dataframe(pd.DataFrame(proxies))
            else:
                st.success("No strong proxy variables detected.")
