"""
Model Insights page — global feature importance, SHAP summary, correlations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data.generate_data import (
    ACTION_LABELS, ACTION_COLORS, ACTION_ICONS, FEATURE_DESCRIPTIONS
)
from utils.model_utils import (
    get_global_feature_importance, preprocess, get_shap_explainer
)


@st.cache_data
def compute_global_shap(_model, _encoders, df: pd.DataFrame, n=300):
    """Compute SHAP values on a sample for global analysis."""
    import shap
    sample = df.sample(min(n, len(df)), random_state=42)
    X, _ = preprocess(sample, encoders=_encoders, fit=False)
    explainer = shap.TreeExplainer(_model)
    shap_vals = explainer.shap_values(X)
    return shap_vals, X


def render(model, encoders, df, X_test, y_test):
    st.markdown("""
    <div class="main-header">
        <h1>📊 Model Insights</h1>
        <p>Global explainability — what features matter most across all farm records.</p>
    </div>
    """, unsafe_allow_html=True)

    from utils.model_utils import ALL_FEATURES

    # ── Global feature importance ──────────────────────────────────────────────
    st.markdown('<p class="section-header">Global Feature Importance (XGBoost Gain)</p>', unsafe_allow_html=True)
    importance_df = get_global_feature_importance(model)
    importance_df["label"] = importance_df["feature"].map(
        lambda f: FEATURE_DESCRIPTIONS.get(f, (f, ""))[0]
    )

    fig = px.bar(
        importance_df.head(12), x="importance", y="label",
        orientation="h",
        color="importance",
        color_continuous_scale=["#d8f3dc", "#40916c", "#1b4332"],
        labels={"importance": "Relative Importance", "label": ""},
        template="plotly_white"
    )
    fig.update_layout(height=380, margin=dict(l=10, r=20, t=10, b=20),
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # ── SHAP global mean |shap| per class ─────────────────────────────────────
    st.markdown('<p class="section-header">Mean |SHAP| Per Feature — By Recommended Action</p>',
                unsafe_allow_html=True)

    with st.spinner("Computing global SHAP values (one-time)…"):
        shap_vals, X_sample = compute_global_shap(model, encoders, df)

    # shap_vals is list[n_classes] of arrays [n_samples, n_features]
    n_classes = len(shap_vals)
    mean_abs = {}
    for cls_id in range(n_classes):
        mean_abs[ACTION_LABELS[cls_id]] = np.abs(shap_vals[cls_id]).mean(axis=0)

    feat_labels = [FEATURE_DESCRIPTIONS.get(f, (f, ""))[0] for f in ALL_FEATURES]
    shap_df = pd.DataFrame(mean_abs, index=feat_labels)

    # Top 10 features by overall importance
    shap_df["total"] = shap_df.sum(axis=1)
    shap_df = shap_df.sort_values("total", ascending=False).head(10).drop("total", axis=1)

    fig2 = go.Figure()
    for cls_id, cls_label in ACTION_LABELS.items():
        if cls_label in shap_df.columns:
            fig2.add_trace(go.Bar(
                name=f"{ACTION_ICONS[cls_id]} {cls_label}",
                x=shap_df.index,
                y=shap_df[cls_label],
                marker_color=ACTION_COLORS[cls_id]
            ))
    fig2.update_layout(
        barmode="group", template="plotly_white",
        height=400, margin=dict(t=10),
        xaxis_tickangle=-30,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Feature correlation heatmap ────────────────────────────────────────────
    st.markdown('<p class="section-header">Feature Correlation Matrix</p>', unsafe_allow_html=True)

    num_cols = [
        "soil_moisture", "soil_ph", "nitrogen_level", "phosphorus_level",
        "potassium_level", "temperature", "rainfall_7d", "humidity",
        "ndvi_score", "pest_pressure_index", "crop_age_days"
    ]
    corr = df[num_cols].corr().round(2)
    readable_cols = [FEATURE_DESCRIPTIONS.get(c, (c, ""))[0] for c in num_cols]

    fig3 = px.imshow(
        corr, x=readable_cols, y=readable_cols,
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        template="plotly_white",
        text_auto=True
    )
    fig3.update_layout(height=500, margin=dict(t=20))
    fig3.update_xaxes(tickangle=45)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Distribution of key features ─────────────────────────────────────────
    st.markdown('<p class="section-header">Feature Distributions by Action</p>', unsafe_allow_html=True)
    feat_to_plot = st.selectbox(
        "Select feature",
        ["soil_moisture", "nitrogen_level", "pest_pressure_index", "ndvi_score", "rainfall_7d"],
        format_func=lambda f: FEATURE_DESCRIPTIONS.get(f, (f, ""))[0]
    )
    plot_df = df.copy()
    plot_df["Action"] = plot_df["recommended_action"].map(
        lambda x: f"{ACTION_ICONS[x]} {ACTION_LABELS[x]}"
    )
    color_map = {
        f"{ACTION_ICONS[k]} {v}": ACTION_COLORS[k]
        for k, v in ACTION_LABELS.items()
    }
    fig4 = px.box(
        plot_df, x="Action", y=feat_to_plot, color="Action",
        color_discrete_map=color_map,
        template="plotly_white",
        labels={feat_to_plot: FEATURE_DESCRIPTIONS.get(feat_to_plot, (feat_to_plot, ""))[0]}
    )
    fig4.update_layout(height=380, showlegend=False, margin=dict(t=10))
    st.plotly_chart(fig4, use_container_width=True)
