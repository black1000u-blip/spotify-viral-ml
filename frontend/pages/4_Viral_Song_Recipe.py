"""
Page 4 — Viral Song Recipe

Shows:
 • Data-driven optimal audio ranges for viral songs (from features.csv)
 • Fallback hardcoded recipe if data unavailable
 • SHAP feature importance images
 • Danceability vs Valence scatter for viral songs
 • Downloadable recipe as JSON
"""

import os
import sys
import json

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
    list_shap_images,
    get_shap_path,
    detect_column,
)
from components.charts import scatter_virality

# ── Heading ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="page-title">🧪 Viral Song Recipe</div>
    <div class="page-subtitle">
        Data-driven dissection of the optimal audio DNA for a chart-topping viral hit.
        Ranges are derived from the IQR of confirmed viral songs in our training set.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Hardcoded fallback recipe ──────────────────────────────────────────────────
FALLBACK_RECIPE = [
    ("Danceability",   "0.75 – 0.90",   "0.82",  92, "High groove factor"),
    ("Energy",         "0.70 – 0.95",   "0.82",  85, "Intense & punchy"),
    ("Valence",        "0.55 – 0.80",   "0.68",  68, "Positive emotional tone"),
    ("Tempo",          "110 – 130 BPM", "120",   65, "Upbeat dancefloor pace"),
    ("Speechiness",    "0.04 – 0.12",   "0.07",  18, "Primarily musical"),
    ("Acousticness",   "0.01 – 0.20",   "0.08",  12, "Electronic / produced"),
]

# Feature config: (col_name, min, max, label, description)
AUDIO_FEATURES = [
    ("danceability",  0.0,  1.0, "Danceability",       "Rhythm suitability for dancing"),
    ("energy",        0.0,  1.0, "Energy",              "Intensity & physical activity"),
    ("valence",       0.0,  1.0, "Valence (Mood)",      "Positiveness of the track"),
    ("speechiness",   0.0,  1.0, "Speechiness",         "Quantity of spoken words"),
    ("acousticness",  0.0,  1.0, "Acousticness",        "Acoustic instrument confidence"),
    ("instrumentalness", 0.0, 1.0, "Instrumentalness",  "Absence of vocals"),
    ("tempo",         50,   220, "Tempo (BPM)",          "Beats per minute"),
    ("loudness",      -60,  0,   "Loudness (dB)",        "Overall mix loudness"),
]


def _recipe_card(
    label: str,
    range_str: str,
    median_str: str,
    pct: int,
    tip: str,
) -> str:
    return f"""
    <div class="recipe-card">
        <div class="recipe-feature-name">{label}</div>
        <div class="recipe-range">{range_str}</div>
        <div style="margin-top:10px;">
            <div class="confidence-bar-container">
                <div class="confidence-bar-fill" style="width:{pct}%;"></div>
            </div>
        </div>
        <div style="font-size:0.75rem; color:#8888aa; margin-top:4px;">
            Median: <span style="color:#f0f0f5; font-family:'JetBrains Mono';">
                {median_str}
            </span>
            &nbsp;·&nbsp; {tip}
        </div>
    </div>
    """


# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading dataset…"):
    df = load_features_csv()

viral_col  = detect_column(df, "viral") if not df.empty else None
viral_df   = df[df[viral_col] == 1] if (viral_col and not df.empty) else pd.DataFrame()
has_data   = not viral_df.empty

# ── Optimal ranges section ─────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">🎯 Optimal Viral Audio Signature</div>',
    unsafe_allow_html=True,
)

if has_data:
    # Live data-driven recipe
    recipe_data = []
    for feat, fmin, fmax, label, tip in AUDIO_FEATURES:
        if feat not in viral_df.columns:
            continue
        q25    = viral_df[feat].quantile(0.25)
        q75    = viral_df[feat].quantile(0.75)
        median = viral_df[feat].median()
        pct    = int(min(100, max(0, (median - fmin) / max(fmax - fmin, 1e-9) * 100)))
        recipe_data.append((label, f"{q25:.2f} – {q75:.2f}", f"{median:.2f}", pct, tip, feat, q25, q75, median))

    cols = st.columns(3)
    for i, (label, rng, med, pct, tip, feat, q25, q75, median) in enumerate(recipe_data):
        with cols[i % 3]:
            st.markdown(_recipe_card(label, rng, med, pct, tip), unsafe_allow_html=True)

    # ── Download recipe JSON ───────────────────────────────────────────────────
    recipe_json = {
        item[0]: {"q25": round(item[6], 4), "median": round(item[8], 4), "q75": round(item[7], 4)}
        for item in recipe_data
    }
    st.download_button(
        label="⬇️  Download Recipe as JSON",
        data=json.dumps(recipe_json, indent=2),
        file_name="viral_audio_recipe.json",
        mime="application/json",
    )

else:
    # Fallback hardcoded recipe
    st.info(
        "ℹ️  `data/processed/features.csv` not loaded or has no `viral` label column. "
        "Showing reference heuristics instead."
    )
    fb_cols = st.columns(3)
    for i, (label, rng, med, pct, tip) in enumerate(FALLBACK_RECIPE):
        with fb_cols[i % 3]:
            st.markdown(_recipe_card(label, rng, med, pct, tip), unsafe_allow_html=True)

# ── Stat comparison: Viral vs Non-Viral ───────────────────────────────────────
if has_data and viral_col:
    nonviral_df = df[df[viral_col] == 0]
    feats_to_compare = [f for f, *_ in AUDIO_FEATURES if f in df.columns]

    if feats_to_compare:
        st.markdown(
            '<div class="section-header">📐 Viral vs Non-Viral — Feature Means</div>',
            unsafe_allow_html=True,
        )
        comp_data = {
            "Feature": feats_to_compare,
            "Viral Mean":     [round(viral_df[f].mean(), 4)    for f in feats_to_compare],
            "Non-Viral Mean": [round(nonviral_df[f].mean(), 4) for f in feats_to_compare],
        }
        comp_df = pd.DataFrame(comp_data)

        import plotly.graph_objects as go
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="Viral", x=comp_df["Feature"], y=comp_df["Viral Mean"],
            marker_color="#1DB954", opacity=0.85,
        ))
        fig_comp.add_trace(go.Bar(
            name="Non-Viral", x=comp_df["Feature"], y=comp_df["Non-Viral Mean"],
            marker_color="#e74c3c", opacity=0.85,
        ))
        fig_comp.update_layout(
            barmode="group",
            title="Mean Feature Values — Viral vs Non-Viral",
            yaxis_title="Mean Value",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,17,24,0.6)",
            font=dict(family="JetBrains Mono", color="#f0f0f5"),
            margin=dict(l=20, r=20, t=48, b=20),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

# ── Viral feature clusters ─────────────────────────────────────────────────────
if has_data and viral_col:
    st.markdown(
        '<div class="section-header">💡 Viral Feature Clusters</div>',
        unsafe_allow_html=True,
    )

    cl1, cl2 = st.columns(2)
    feat_options = [f for f, *_ in AUDIO_FEATURES if f in df.columns]

    with cl1:
        x1 = st.selectbox(
            "Scatter X", feat_options,
            index=feat_options.index("danceability") if "danceability" in feat_options else 0,
            key="cluster_x",
        )
    with cl2:
        y1 = st.selectbox(
            "Scatter Y", feat_options,
            index=feat_options.index("valence") if "valence" in feat_options else 1,
            key="cluster_y",
        )

    fig_clust = scatter_virality(df, x1, y1, viral_col)
    st.plotly_chart(fig_clust, use_container_width=True)

# ── SHAP Feature Importance ────────────────────────────────────────────────────
shap_images = list_shap_images()
if shap_images:
    st.markdown(
        '<div class="section-header">🔍 SHAP Feature Importance</div>',
        unsafe_allow_html=True,
    )

    # Tabs: one per SHAP image
    tabs_needed = min(len(shap_images), 6)
    tab_labels  = [f.replace("_", " ").replace(".png", "").title()
                   for f in shap_images[:tabs_needed]]
    shap_tabs   = st.tabs(tab_labels)

    for tab, fname in zip(shap_tabs, shap_images[:tabs_needed]):
        path = get_shap_path(fname)
        if os.path.exists(path):
            with tab:
                st.image(path, use_column_width=True)
        else:
            with tab:
                st.warning(f"Image not found: {fname}")
else:
    st.info(
        "ℹ️  No SHAP images found in `models/shap/`. "
        "Run the SHAP analysis step to generate them."
    )

# ── Pro tips ──────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">💬 Producer Tips</div>',
    unsafe_allow_html=True,
)

tips = [
    ("🎯", "Aim for 120–128 BPM", "This range dominates pop & EDM charts."),
    ("🔊", "Keep loudness −6 to −3 dB", "Loud masters cut through streaming playlists."),
    ("💃", "Danceability above 0.75", "High groove factor correlates strongly with virality."),
    ("☀️", "Valence ≥ 0.60", "Positive, euphoric tracks get shared more."),
    ("🎤", "Speechiness 0.04–0.12", "Mostly musical with catchy hooks wins."),
    ("⚡", "Energy ≥ 0.70", "High-energy releases perform better on social platforms."),
]

tip_c1, tip_c2, tip_c3 = st.columns(3)
tip_cols = [tip_c1, tip_c2, tip_c3]
for i, (emoji, title, desc) in enumerate(tips):
    with tip_cols[i % 3]:
        st.markdown(
            f"""
            <div class="recipe-card" style="border-left-color:#00d4ff;">
                <div style="font-size:1.6rem; margin-bottom:6px;">{emoji}</div>
                <div style="font-family:'Syne',sans-serif; font-weight:700;
                             font-size:0.95rem; color:#f0f0f5; margin-bottom:4px;">
                    {title}
                </div>
                <div style="font-size:0.78rem; color:#8888aa;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
