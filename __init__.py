"""
Explanation page — SHAP waterfall, force plot, and feature contribution breakdown.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from data.generate_data import (
    ACTION_LABELS, ACTION_COLORS, ACTION_ICONS, FEATURE_DESCRIPTIONS
)
from utils.model_utils import explain_single_prediction
from data.generate_data import CROPS, SOIL_TYPES, REGIONS


def shap_waterfall(feature_shap: dict, pred_class: int, base_value: float = 0.0):
    """Render a SHAP waterfall chart using Plotly."""
    # Sort by absolute SHAP value, take top 12
    sorted_feats = sorted(feature_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
    sorted_feats = sorted_feats[::-1]  # bottom to top

    labels = []
    values = []
    for feat, val in sorted_feats:
        desc = FEATURE_DESCRIPTIONS.get(feat, (feat, ""))[0]
        labels.append(desc)
        values.append(val)

    colors = ["#E53935" if v > 0 else "#1E88E5" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in values],
        textposition="outside",
    ))

    action_color = ACTION_COLORS[pred_class]
    fig.update_layout(
        title=f"SHAP Feature Contributions → {ACTION_ICONS[pred_class]} {ACTION_LABELS[pred_class]}",
        xaxis_title="SHAP Value (impact on model output)",
        yaxis_title="",
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=80, t=60, b=40),
        shapes=[dict(
            type="line", x0=0, x1=0,
            y0=-0.5, y1=len(labels) - 0.5,
            line=dict(color="#333", width=1.5)
        )]
    )
    return fig


def render(model, encoders):
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Explain Prediction</h1>
        <p>Understand exactly why the AI made its recommendation using SHAP (SHapley Additive exPlanations).</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <b>What is SHAP?</b> SHAP values tell you how much each feature
        <span style="color:#E53935;font-weight:600">↑ pushed the prediction higher</span> or
        <span style="color:#1E88E5;font-weight:600">↓ pushed it lower</span>
        for a specific field. This makes AI decisions transparent and auditable.
    </div>
    """, unsafe_allow_html=True)

    # ── Check if we have a prediction from the recommendation page ────────────
    has_prev = "last_input" in st.session_state

    tab1, tab2 = st.tabs(["📋 Use Last Recommendation" if has_prev else "📋 Quick Example", "⚙️ Custom Input"])

    with tab1:
        if has_prev:
            input_dict = st.session_state["last_input"]
            pred_class = st.session_state["last_pred"]
            pred_proba = st.session_state["last_proba"]
            feature_shap = st.session_state["last_shap"]
            st.success("Using the field conditions from your last recommendation.")
        else:
            # Default demo case: irrigation needed
            input_dict = {
                "soil_moisture": 18.0, "soil_ph": 6.2, "nitrogen_level": 60.0,
                "phosphorus_level": 28.0, "potassium_level": 95.0, "organic_matter": 3.5,
                "temperature": 30.0, "rainfall_7d": 3.0, "humidity": 38.0,
                "solar_radiation": 22.0, "wind_speed": 12.0, "crop_age_days": 45,
                "leaf_area_index": 2.8, "ndvi_score": 0.58, "pest_pressure_index": 1.5,
                "crop_type": "Wheat", "soil_type": "Sandy", "region": "North America",
            }
            with st.spinner("Running demo prediction…"):
                pred_class, pred_proba, feature_shap, _ = explain_single_prediction(
                    model, encoders, input_dict
                )
            st.info("Showing a demo prediction. Go to 🔮 Get Recommendation first to explain your own inputs.")

        _render_explanation(pred_class, pred_proba, feature_shap, input_dict)

    with tab2:
        st.markdown("Enter a specific field scenario to explain:")
        with st.form("explain_custom"):
            c1, c2, c3 = st.columns(3)
            with c1:
                crop_type = st.selectbox("Crop Type", CROPS, key="ex_crop")
                soil_moisture_e = st.slider("Soil Moisture (%)", 0.0, 100.0, 30.0, key="ex_sm")
                nitrogen_level_e = st.slider("Nitrogen (kg/ha)", 0.0, 150.0, 20.0, key="ex_n")
            with c2:
                soil_type = st.selectbox("Soil Type", SOIL_TYPES, key="ex_st")
                temperature_e = st.slider("Temperature (°C)", -5.0, 45.0, 28.0, key="ex_t")
                rainfall_e = st.slider("7-Day Rainfall (mm)", 0.0, 120.0, 5.0, key="ex_r")
            with c3:
                region = st.selectbox("Region", REGIONS, key="ex_reg")
                ndvi_e = st.slider("NDVI Score", 0.0, 1.0, 0.5, key="ex_ndvi")
                pest_e = st.slider("Pest Pressure (0-10)", 0.0, 10.0, 7.5, key="ex_pest")

            run = st.form_submit_button("🔍 Explain This Scenario", use_container_width=True)

        if run:
            custom_input = {
                "soil_moisture": soil_moisture_e, "soil_ph": 6.5,
                "nitrogen_level": nitrogen_level_e, "phosphorus_level": 25.0,
                "potassium_level": 90.0, "organic_matter": 3.0,
                "temperature": temperature_e, "rainfall_7d": rainfall_e,
                "humidity": 60.0, "solar_radiation": 18.0, "wind_speed": 10.0,
                "crop_age_days": 50, "leaf_area_index": 3.0,
                "ndvi_score": ndvi_e, "pest_pressure_index": pest_e,
                "crop_type": crop_type, "soil_type": soil_type, "region": region,
            }
            with st.spinner("Computing SHAP values…"):
                p_class, p_proba, f_shap, _ = explain_single_prediction(model, encoders, custom_input)
            _render_explanation(p_class, p_proba, f_shap, custom_input)


def _render_explanation(pred_class, pred_proba, feature_shap, input_dict):
    """Shared rendering for waterfall + feature table."""
    color = ACTION_COLORS[pred_class]
    icon = ACTION_ICONS[pred_class]
    label = ACTION_LABELS[pred_class]

    # Prediction banner
    st.markdown(f"""
    <div style="background:white;border-radius:12px;padding:1.2rem 1.5rem;
                border-left:5px solid {color};margin:1rem 0;">
        <span style="color:#6c757d;font-size:0.8rem;">Predicted Action</span><br>
        <span style="font-size:1.6rem;font-weight:800;color:{color};">{icon} {label}</span>
        <span style="float:right;font-size:1.1rem;color:{color};font-weight:700;">
            {pred_proba[pred_class]*100:.1f}% confident
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Waterfall chart
    st.markdown('<p class="section-header">Feature Contributions (SHAP Waterfall)</p>', unsafe_allow_html=True)
    fig = shap_waterfall(feature_shap, pred_class)
    st.plotly_chart(fig, use_container_width=True)

    # Legend
    st.markdown("""
    <div style="display:flex;gap:1.5rem;font-size:0.85rem;margin-bottom:1rem;">
        <span>🔴 <b>Red bars</b>: Feature pushed prediction toward this action</span>
        <span>🔵 <b>Blue bars</b>: Feature pushed prediction away from this action</span>
    </div>
    """, unsafe_allow_html=True)

    # Top drivers table
    st.markdown('<p class="section-header">Top Decision Drivers</p>', unsafe_allow_html=True)
    sorted_feats = sorted(feature_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    rows = []
    for feat, shap_val in sorted_feats:
        desc, detail = FEATURE_DESCRIPTIONS.get(feat, (feat, ""))
        raw_val = input_dict.get(feat, "N/A")
        direction = "▲ Supports" if shap_val > 0 else "▼ Against"
        dir_color = "#E53935" if shap_val > 0 else "#1E88E5"
        rows.append({
            "Feature": desc,
            "Your Value": f"{raw_val:.2f}" if isinstance(raw_val, float) else str(raw_val),
            "SHAP Impact": f"{shap_val:+.4f}",
            "Direction": direction,
        })

    tbl = pd.DataFrame(rows)
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    # Plain-language summary
    top_push = sorted(feature_shap.items(), key=lambda x: x[1], reverse=True)[:2]
    top_pull = sorted(feature_shap.items(), key=lambda x: x[1])[:2]

    push_names = " and ".join(FEATURE_DESCRIPTIONS.get(f, (f, ""))[0] for f, _ in top_push)
    pull_names = " and ".join(FEATURE_DESCRIPTIONS.get(f, (f, ""))[0] for f, _ in top_pull)

    st.markdown(f"""
    <div class="info-box" style="margin-top:1rem;">
        <b>📝 Plain-Language Explanation:</b><br>
        The AI recommended <b>{icon} {label}</b> mainly because
        <span style="color:#E53935;font-weight:600">{push_names}</span> strongly
        support this action. Factors like
        <span style="color:#1E88E5;font-weight:600">{pull_names}</span>
        worked against this recommendation but were outweighed.
    </div>
    """, unsafe_allow_html=True)
