"""
app.py — Main entry point for the Spotify Virality Prediction Dashboard.

Run from the project root:
    streamlit run frontend/app.py
"""

import os
import sys

import streamlit as st

# ── Path bootstrap ─────────────────────────────────────────────────────────────
# Ensure `frontend/` is importable so all pages can do `from components.xxx`
sys.path.insert(0, os.path.dirname(__file__))

# ── Page config (MUST be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Spotify Virality · AI Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject global CSS ──────────────────────────────────────────────────────────
_css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(_css_path):
    with open(_css_path) as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# ── Imports ────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np

from components.model_loader import (
    load_raw_csv,
    load_features_csv,
    load_model_results,
    get_viz_path,
    detect_column,
)
from components.charts import popularity_distribution, correlation_heatmap

# ── Sidebar branding ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 20px 0 28px;">
            <div style="font-size:2.8rem; line-height:1;">🎵</div>
            <div style="
                font-family:'Syne',sans-serif;
                font-size:1.15rem;
                font-weight:800;
                background:linear-gradient(135deg,#1DB954,#00d4ff);
                -webkit-background-clip:text;
                -webkit-text-fill-color:transparent;
                margin-top:8px;
            ">Spotify Virality</div>
            <div style="
                font-size:0.68rem;
                color:#8888aa;
                letter-spacing:0.18em;
                text-transform:uppercase;
                margin-top:4px;
            ">AI Prediction System</div>
        </div>
        <hr style="border:none; border-top:1px solid rgba(255,255,255,0.07); margin-bottom:20px;">
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-size:0.68rem; color:#8888aa; text-transform:uppercase; "
        "letter-spacing:0.13em; margin-bottom:10px;'>Navigation</div>",
        unsafe_allow_html=True,
    )
    st.page_link("app.py",                         label="🏠  Home",               )
    st.page_link("pages/1_Predict_Virality.py",    label="🎤  Predict Virality",   )
    st.page_link("pages/2_Data_Insights.py",       label="📊  Data Insights",      )
    st.page_link("pages/3_Model_Performance.py",   label="🏆  Model Performance",  )
    st.page_link("pages/4_Viral_Song_Recipe.py",   label="🧪  Viral Song Recipe",  )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.68rem; color:#555570; text-align:center;'>"
        "Built with Streamlit + Plotly</div>",
        unsafe_allow_html=True,
    )

# ── Page title ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="page-title">🎵 Spotify Viral Hit Prediction System</div>
    <div class="page-subtitle">
        End-to-end machine learning pipeline that predicts whether a song will go
        viral based on its audio DNA — powered by a Stacking Ensemble of
        CatBoost · LightGBM · XGBoost · Random Forest · Logistic Regression.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading datasets…"):
    raw_df       = load_raw_csv()
    feat_df      = load_features_csv()
    model_results = load_model_results()

# ── KPI Cards ──────────────────────────────────────────────────────────────────
n_songs    = f"{len(raw_df):,}"  if not raw_df.empty       else "—"
n_features = len(feat_df.columns) if not feat_df.empty     else "—"

best_model = "—"
best_acc   = "—"
n_models   = "—"

if not model_results.empty:
    acc_col   = detect_column(model_results, "acc")
    model_col = model_results.columns[0]
    n_models  = str(len(model_results))
    if acc_col:
        idx        = model_results[acc_col].idxmax()
        best_model = model_results.loc[idx, model_col]
        best_acc   = f"{model_results.loc[idx, acc_col]:.4f}"

kpis = [
    ("🎵", n_songs,    "Songs Analyzed"),
    ("⚙️", n_features, "Engineered Features"),
    ("🤖", n_models,   "Models Trained"),
    ("🏆", best_acc,   f"Best Accuracy · {best_model}"),
]

cols = st.columns(4)
for col, (icon, value, label) in zip(cols, kpis):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size:1.8rem; margin-bottom:8px;">{icon}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Static visualisations ──────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">📈 Dataset Visualisations</div>',
    unsafe_allow_html=True,
)

viz_files = [
    ("correlation_matrix.png",      "Correlation Matrix"),
    ("feature_distributions.png",   "Feature Distributions"),
    ("popularity_distribution.png", "Popularity Distribution"),
    ("viral_vs_nonviral.png",       "Viral vs Non-Viral"),
]

viz_cols = st.columns(2)
rendered = 0
for fname, title in viz_files:
    path = get_viz_path(fname)
    if os.path.exists(path):
        with viz_cols[rendered % 2]:
            st.markdown(f"**{title}**")
            st.image(path, use_column_width=True)
        rendered += 1

if rendered == 0:
    st.info("ℹ️ No static visualisation images found in `visualizations/` yet.")

# ── Interactive popularity chart ───────────────────────────────────────────────
if not raw_df.empty:
    pop_col = detect_column(raw_df, "popular")
    if pop_col:
        st.markdown(
            '<div class="section-header">📊 Interactive Popularity Distribution</div>',
            unsafe_allow_html=True,
        )
        fig = popularity_distribution(raw_df, pop_col)
        st.plotly_chart(fig, use_container_width=True)

# ── Interactive correlation heatmap ───────────────────────────────────────────
if not feat_df.empty:
    num_cols = feat_df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) >= 2:
        st.markdown(
            '<div class="section-header">🔗 Feature Correlation Heatmap</div>',
            unsafe_allow_html=True,
        )
        fig = correlation_heatmap(feat_df, num_cols[:14])
        st.plotly_chart(fig, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-top:60px; text-align:center; color:#555570; font-size:0.74rem;
                border-top:1px solid rgba(255,255,255,0.06); padding-top:20px;">
        Spotify Virality Prediction System &nbsp;·&nbsp;
        Built with Streamlit &amp; Plotly &nbsp;·&nbsp;
        Stacking Ensemble ML
    </div>
    """,
    unsafe_allow_html=True,
)
