"""
Overview page — summary stats, action distribution, key metrics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data.generate_data import ACTION_LABELS, ACTION_COLORS, ACTION_ICONS

PALETTE = list(ACTION_COLORS.values())


def render(df: pd.DataFrame, accuracy: float, report: dict):
    st.markdown("""
    <div class="main-header">
        <h1>🌾 Farm AI — Explainable Agronomy Recommendations</h1>
        <p>AI-powered farm management decisions with full SHAP explainability · Jacob AI R&D Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <h3>Model Accuracy</h3>
            <div class="value">{accuracy*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <h3>Total Farm Records</h3>
            <div class="value">{len(df):,}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        top_action = df["recommended_action"].value_counts().idxmax()
        st.markdown(f"""<div class="metric-card">
            <h3>Most Common Action</h3>
            <div class="value">{ACTION_ICONS[top_action]} {ACTION_LABELS[top_action]}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        avg_ndvi = df["ndvi_score"].mean()
        st.markdown(f"""<div class="metric-card">
            <h3>Avg NDVI Score</h3>
            <div class="value">{avg_ndvi:.2f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Action Distribution</p>', unsafe_allow_html=True)
        action_counts = df["recommended_action"].value_counts().reset_index()
        action_counts.columns = ["action", "count"]
        action_counts["label"] = action_counts["action"].map(ACTION_LABELS)
        action_counts["color"] = action_counts["action"].map(ACTION_COLORS)
        action_counts["icon"] = action_counts["action"].map(ACTION_ICONS)

        fig = px.bar(
            action_counts, x="label", y="count",
            color="label",
            color_discrete_sequence=PALETTE,
            labels={"label": "", "count": "Records"},
            template="plotly_white"
        )
        fig.update_layout(showlegend=False, height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Action by Crop Type</p>', unsafe_allow_html=True)
        pivot = df.groupby(["crop_type", "recommended_action"]).size().reset_index(name="count")
        pivot["action_label"] = pivot["recommended_action"].map(ACTION_LABELS)
        fig2 = px.bar(
            pivot, x="crop_type", y="count", color="action_label",
            color_discrete_sequence=PALETTE,
            labels={"crop_type": "", "count": "Records", "action_label": "Action"},
            template="plotly_white", barmode="stack"
        )
        fig2.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Model performance table ───────────────────────────────────────────────
    st.markdown('<p class="section-header">Per-Class Model Performance</p>', unsafe_allow_html=True)
    rows = []
    for cls_id, label in ACTION_LABELS.items():
        if str(cls_id) in report:
            m = report[str(cls_id)]
            rows.append({
                "Action": f"{ACTION_ICONS[cls_id]} {label}",
                "Precision": f"{m['precision']:.3f}",
                "Recall": f"{m['recall']:.3f}",
                "F1 Score": f"{m['f1-score']:.3f}",
                "Support": int(m["support"])
            })
    perf_df = pd.DataFrame(rows)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    # ── Scatter: soil moisture vs NDVI ───────────────────────────────────────
    st.markdown('<p class="section-header">Soil Moisture vs NDVI by Action</p>', unsafe_allow_html=True)
    sample = df.sample(400, random_state=1)
    sample["Action"] = sample["recommended_action"].map(lambda x: f"{ACTION_ICONS[x]} {ACTION_LABELS[x]}")
    fig3 = px.scatter(
        sample, x="soil_moisture", y="ndvi_score", color="Action",
        color_discrete_sequence=PALETTE,
        labels={"soil_moisture": "Soil Moisture (%)", "ndvi_score": "NDVI Score"},
        opacity=0.7, template="plotly_white"
    )
    fig3.update_layout(height=350, margin=dict(t=10))
    st.plotly_chart(fig3, use_container_width=True)
