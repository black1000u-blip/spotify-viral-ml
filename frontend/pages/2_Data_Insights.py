"""
Page 2 — Data Insights

Shows interactive Plotly charts built from features.csv, plus
static visualisation images from the visualizations/ folder.
"""

import os
import sys

import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── CSS ────────────────────────────────────────────────────────────────────────
_css = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
if os.path.exists(_css):
    with open(_css) as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

from components.model_loader import (
    load_features_csv,
    load_raw_csv,
    get_viz_path,
    detect_column,
)
from components.charts import (
    scatter_virality,
    tempo_histogram,
    correlation_heatmap,
    popularity_distribution,
)

# ── Heading ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="page-title">📊 Data Insights</div>
    <div class="page-subtitle">
        Deep-dive into audio feature patterns across viral and non-viral songs.
        All charts are interactive — hover, zoom, and filter freely.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading dataset…"):
    feat_df = load_features_csv()
    raw_df  = load_raw_csv()

# Detect label column
viral_col = detect_column(feat_df, "viral") if not feat_df.empty else None

# ── Dataset overview ───────────────────────────────────────────────────────────
if not feat_df.empty:
    st.markdown(
        '<div class="section-header">🗂️ Dataset Overview</div>',
        unsafe_allow_html=True,
    )

    ov1, ov2, ov3, ov4 = st.columns(4)
    n_rows = len(feat_df)
    n_cols = len(feat_df.columns)
    n_viral    = int(feat_df[viral_col].sum())   if viral_col else "—"
    n_nonviral = n_rows - n_viral if isinstance(n_viral, int) else "—"
    viral_pct  = f"{n_viral/n_rows*100:.1f}%" if isinstance(n_viral, int) else "—"

    for col_widget, icon, val, lbl in [
        (ov1, "📋", f"{n_rows:,}", "Total Samples"),
        (ov2, "⚙️", n_cols,       "Feature Columns"),
        (ov3, "🔥", n_viral,      "Viral Songs"),
        (ov4, "📉", n_nonviral,   "Non-Viral Songs"),
    ]:
        with col_widget:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:1.6rem; margin-bottom:6px;">{icon}</div>
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{lbl}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

# ── Interactive scatter ────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">🔬 Interactive Feature Scatter</div>',
    unsafe_allow_html=True,
)

num_cols = feat_df.select_dtypes(include=np.number).columns.tolist() if not feat_df.empty else []
default_x = "danceability" if "danceability" in num_cols else (num_cols[0] if num_cols else None)
default_y = "energy"       if "energy"       in num_cols else (num_cols[1] if len(num_cols) > 1 else None)

if num_cols and default_x and default_y:
    sc1, sc2 = st.columns(2)
    with sc1:
        x_axis = st.selectbox("X-axis feature", num_cols,
                               index=num_cols.index(default_x))
    with sc2:
        y_axis = st.selectbox("Y-axis feature", num_cols,
                               index=num_cols.index(default_y))

    fig_scatter = scatter_virality(feat_df, x_axis, y_axis, viral_col or num_cols[0])
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.warning("⚠️  features.csv not found or has no numeric columns.")

# ── Tempo histogram ────────────────────────────────────────────────────────────
if not feat_df.empty and "tempo" in feat_df.columns:
    st.markdown(
        '<div class="section-header">🥁 Tempo Distribution</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(tempo_histogram(feat_df, viral_col), use_container_width=True)

# ── Correlation heatmap ────────────────────────────────────────────────────────
if not feat_df.empty and len(num_cols) >= 2:
    st.markdown(
        '<div class="section-header">🔗 Correlation Heatmap</div>',
        unsafe_allow_html=True,
    )
    fig_heat = correlation_heatmap(feat_df, num_cols[:14])
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Popularity from raw data ───────────────────────────────────────────────────
if not raw_df.empty:
    pop_col = detect_column(raw_df, "popular")
    if pop_col:
        st.markdown(
            '<div class="section-header">🎯 Popularity Score Distribution</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(popularity_distribution(raw_df, pop_col), use_container_width=True)

# ── Viral class balance bar ────────────────────────────────────────────────────
if not feat_df.empty and viral_col and isinstance(n_viral, int):
    import plotly.graph_objects as go
    st.markdown(
        '<div class="section-header">⚖️ Class Balance</div>',
        unsafe_allow_html=True,
    )
    fig_bal = go.Figure(go.Bar(
        x=["Non-Viral", "Viral"],
        y=[n_nonviral, n_viral],
        marker_color=["#e74c3c", "#1DB954"],
        text=[f"{n_nonviral:,}", f"{n_viral:,}"],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=12),
        opacity=0.85,
    ))
    fig_bal.update_layout(
        title="Viral vs Non-Viral Class Count",
        yaxis_title="Count",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,17,24,0.6)",
        font=dict(family="JetBrains Mono", color="#f0f0f5"),
        margin=dict(l=20, r=20, t=48, b=20),
    )
    st.plotly_chart(fig_bal, use_container_width=True)

# ── Raw data preview ───────────────────────────────────────────────────────────
with st.expander("📋 Preview features.csv (first 100 rows)"):
    if not feat_df.empty:
        st.dataframe(feat_df.head(100), use_container_width=True)
    else:
        st.info("features.csv not loaded.")

# ── Static visualisation images ────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">🖼️ Static Visualisations</div>',
    unsafe_allow_html=True,
)

static_imgs = [
    ("viral_vs_nonviral.png",      "Viral vs Non-Viral"),
    ("threshold_optimization.png", "Threshold Optimisation"),
    ("feature_distributions.png",  "Feature Distributions"),
    ("popularity_distribution.png","Popularity Distribution"),
]

sv_cols = st.columns(2)
rendered = 0
for fname, title in static_imgs:
    path = get_viz_path(fname)
    if os.path.exists(path):
        with sv_cols[rendered % 2]:
            st.markdown(f"**{title}**")
            st.image(path, use_column_width=True)
        rendered += 1

if rendered == 0:
    st.info("No static images found in `visualizations/`.")
