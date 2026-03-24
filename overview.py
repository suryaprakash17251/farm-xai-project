"""
Farm AI — Explainable AI Dashboard for Agronomy Recommendations
Main entry point for Streamlit app.
"""

import streamlit as st
import sys
from pathlib import Path

# Make sure imports work regardless of run location
sys.path.insert(0, str(Path(__file__).parent))

from utils.model_utils import train_model, load_model
from data.generate_data import generate_farm_data, ACTION_LABELS, ACTION_ICONS

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Farm AI — Explainable Recommendations",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #f4f6f0; }
    .main-header {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 50%, #40916c 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p { color: rgba(255,255,255,0.85); margin: 0.3rem 0 0; font-size: 1rem; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border-left: 4px solid #40916c;
    }
    .metric-card h3 { margin: 0; font-size: 0.8rem; color: #6c757d; text-transform: uppercase; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #1b4332; }
    .action-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 24px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .info-box {
        background: #e8f5e9;
        border: 1px solid #a5d6a7;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    [data-testid="stSidebar"] { background-color: #1b4332; }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label { color: #d8f3dc !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #b7e4c7 !important; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1b4332;
        border-bottom: 2px solid #40916c;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training AI model on synthetic farm data…")
def get_model_and_data():
    df = generate_farm_data(2000)
    model, encoders, acc, report, X_test, y_test = train_model(df)
    return model, encoders, acc, report, df, X_test, y_test

model, encoders, accuracy, report, df, X_test, y_test = get_model_and_data()

# ─── Sidebar navigation ───────────────────────────────────────────────────────
st.sidebar.markdown("## 🌾 Farm AI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "🔮 Get Recommendation", "🧠 Explain Prediction", "📊 Model Insights", "📋 Dataset Explorer"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model Accuracy:** {accuracy*100:.1f}%")
st.sidebar.markdown(f"**Training samples:** 1,600")
st.sidebar.markdown(f"**Test samples:** 400")
st.sidebar.markdown(f"**Features:** 18")
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='color:#b7e4c7;font-size:0.75rem;'>
Built with XGBoost + SHAP<br>
Jacob AI — Agri-Tech R&D<br>
</div>
""", unsafe_allow_html=True)

# ─── Route pages ──────────────────────────────────────────────────────────────
if "Overview" in page:
    from pages import overview
    overview.render(df, accuracy, report)

elif "Get Recommendation" in page:
    from pages import recommendation
    recommendation.render(model, encoders)

elif "Explain Prediction" in page:
    from pages import explanation
    explanation.render(model, encoders)

elif "Model Insights" in page:
    from pages import insights
    insights.render(model, encoders, df, X_test, y_test)

elif "Dataset Explorer" in page:
    from pages import explorer
    explorer.render(df)
