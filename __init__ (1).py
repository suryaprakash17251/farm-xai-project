"""
Recommendation page — user inputs farm conditions, gets AI recommendation.
"""

import streamlit as st
import numpy as np
from data.generate_data import (
    ACTION_LABELS, ACTION_COLORS, ACTION_ICONS,
    CROPS, SOIL_TYPES, REGIONS, FEATURE_DESCRIPTIONS
)
from utils.model_utils import explain_single_prediction


def render(model, encoders):
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Get Farm Recommendation</h1>
        <p>Enter your field conditions and get an AI-powered action recommendation with confidence scores.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <b>How it works:</b> Enter your current farm sensor readings below. The XGBoost model
        will predict the best agronomic action along with confidence for each possible action.
    </div>
    """, unsafe_allow_html=True)

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("farm_inputs"):
        st.markdown('<p class="section-header">🌱 Crop Information</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            crop_type = st.selectbox("Crop Type", CROPS)
        with c2:
            region = st.selectbox("Region", REGIONS)
        with c3:
            crop_age_days = st.slider("Crop Age (days)", 0, 180, 60)

        st.markdown('<p class="section-header">🪨 Soil Conditions</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            soil_type = st.selectbox("Soil Type", SOIL_TYPES)
            soil_moisture = st.slider("Soil Moisture (%)", 0.0, 100.0, 45.0, step=0.5)
        with c2:
            soil_ph = st.slider("Soil pH", 4.5, 8.5, 6.5, step=0.1)
            organic_matter = st.slider("Organic Matter (%)", 0.0, 10.0, 3.0, step=0.1)
        with c3:
            nitrogen_level = st.slider("Nitrogen (kg/ha)", 0.0, 150.0, 50.0, step=1.0)
            phosphorus_level = st.slider("Phosphorus (kg/ha)", 0.0, 80.0, 25.0, step=1.0)
        potassium_level = st.slider("Potassium (kg/ha)", 0.0, 250.0, 90.0, step=1.0)

        st.markdown('<p class="section-header">⛅ Weather Conditions</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            temperature = st.slider("Temperature (°C)", -5.0, 45.0, 22.0, step=0.5)
            rainfall_7d = st.slider("7-Day Rainfall (mm)", 0.0, 120.0, 15.0, step=0.5)
        with c2:
            humidity = st.slider("Humidity (%)", 20.0, 95.0, 60.0, step=1.0)
            solar_radiation = st.slider("Solar Radiation (MJ/m²/day)", 5.0, 30.0, 18.0, step=0.5)
        with c3:
            wind_speed = st.slider("Wind Speed (km/h)", 0.0, 40.0, 8.0, step=0.5)

        st.markdown('<p class="section-header">📡 Remote Sensing</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            ndvi_score = st.slider("NDVI Score", 0.0, 1.0, 0.65, step=0.01)
        with c2:
            leaf_area_index = st.slider("Leaf Area Index", 0.0, 6.0, 3.0, step=0.1)
        with c3:
            pest_pressure_index = st.slider("Pest Pressure Index (0–10)", 0.0, 10.0, 2.0, step=0.1)

        submitted = st.form_submit_button("🚀 Get AI Recommendation", use_container_width=True)

    if submitted:
        input_dict = {
            "soil_moisture": soil_moisture,
            "soil_ph": soil_ph,
            "nitrogen_level": nitrogen_level,
            "phosphorus_level": phosphorus_level,
            "potassium_level": potassium_level,
            "organic_matter": organic_matter,
            "temperature": temperature,
            "rainfall_7d": rainfall_7d,
            "humidity": humidity,
            "solar_radiation": solar_radiation,
            "wind_speed": wind_speed,
            "crop_age_days": crop_age_days,
            "leaf_area_index": leaf_area_index,
            "ndvi_score": ndvi_score,
            "pest_pressure_index": pest_pressure_index,
            "crop_type": crop_type,
            "soil_type": soil_type,
            "region": region,
        }

        with st.spinner("Running AI inference…"):
            pred_class, pred_proba, feature_shap, X_proc = explain_single_prediction(
                model, encoders, input_dict
            )

        # ── Result card ───────────────────────────────────────────────────────
        color = ACTION_COLORS[pred_class]
        icon = ACTION_ICONS[pred_class]
        label = ACTION_LABELS[pred_class]
        confidence = pred_proba[pred_class] * 100

        st.markdown("---")
        st.markdown(f"""
        <div style="background:white;border-radius:14px;padding:2rem;
                    border-left:6px solid {color};box-shadow:0 2px 8px rgba(0,0,0,0.1);
                    margin-bottom:1rem;">
            <div style="font-size:0.85rem;color:#6c757d;margin-bottom:0.3rem;">
                AI Recommendation — {confidence:.1f}% confidence
            </div>
            <div style="font-size:2.2rem;font-weight:800;color:{color};">
                {icon} {label}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability bars for all actions
        st.markdown('<p class="section-header">Confidence for All Actions</p>', unsafe_allow_html=True)
        for cls_id, prob in enumerate(pred_proba):
            lbl = f"{ACTION_ICONS[cls_id]} {ACTION_LABELS[cls_id]}"
            col = ACTION_COLORS[cls_id]
            is_pred = cls_id == pred_class
            border = f"border:2px solid {col}" if is_pred else "border:1px solid #dee2e6"
            st.markdown(f"""
            <div style="background:white;border-radius:8px;padding:0.7rem 1rem;
                        margin-bottom:0.5rem;{border};">
                <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">
                    <span style="font-weight:{'700' if is_pred else '400'}">{lbl}</span>
                    <span style="color:{col};font-weight:700">{prob*100:.1f}%</span>
                </div>
                <div style="background:#f0f0f0;border-radius:4px;height:8px;">
                    <div style="background:{col};width:{prob*100:.1f}%;height:8px;
                                border-radius:4px;transition:width 0.5s;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Save to session state for explain page
        st.session_state["last_input"] = input_dict
        st.session_state["last_pred"] = pred_class
        st.session_state["last_proba"] = pred_proba
        st.session_state["last_shap"] = feature_shap

        st.info("💡 Go to **🧠 Explain Prediction** in the sidebar to see WHY the AI made this decision.")
