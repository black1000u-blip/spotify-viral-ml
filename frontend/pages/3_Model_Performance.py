"""
Page 3 — Model Performance

Displays:
 • Classification model medal leaderboard
 • Multi-metric bar chart
 • Interactive accuracy bar
 • Cross-validation box plot
 • Regression model table + bar chart
 • Static comparison images
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
    load_model_results,
    load_regression_results,
    load_cv_results,
    load_optimal_thresholds,
    get_viz_path,
    detect_column,
)
from components.charts import (
    model_accuracy_bar,
    multi_metric_bar,
    cv_boxplot,
    regression_bar,
)

# ── Heading ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="page-title">🏆 Model Performance</div>
    <div class="page-subtitle">
        Side-by-side comparison of all trained classifiers and regressors —
        accuracy, F1, AUC, cross-validation stability, and optimal thresholds.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading model results…"):
    model_df   = load_model_results()
    reg_df     = load_regression_results()
    cv_data    = load_cv_results()
    thresholds = load_optimal_thresholds()

# ── Classification Leaderboard ─────────────────────────────────────────────────
if not model_df.empty:
    st.markdown(
        '<div class="section-header">🥇 Classification Model Leaderboard</div>',
        unsafe_allow_html=True,
    )

    acc_col   = detect_column(model_df, "acc")
    model_col = model_df.columns[0]

    if acc_col:
        sorted_df = model_df.sort_values(acc_col, ascending=False).reset_index(drop=True)
        medals = ["🥇", "🥈", "🥉"]

        for i, row in sorted_df.iterrows():
            medal = medals[i] if i < 3 else f"#{i+1}"
            # Build extra metric pills
            extra_metrics = ""
            for c in [c for c in model_df.columns if c not in [model_col, acc_col]]:
                val = row[c]
                if isinstance(val, float):
                    extra_metrics += (
                        f"<span style='font-size:0.75rem; color:#8888aa; "
                        f"margin-left:14px; font-family:JetBrains Mono;'>"
                        f"{c}: <span style='color:#00d4ff;'>{val:.4f}</span></span>"
                    )

            st.markdown(
                f"""
                <div class="leaderboard-row">
                    <span style="font-size:1.3rem; min-width:32px;">{medal}</span>
                    <span style="font-family:'Syne',sans-serif; font-weight:700;
                                 font-size:1rem; flex:1; margin-left:12px;">
                        {row[model_col]}
                    </span>
                    {extra_metrics}
                    <span style="font-family:'JetBrains Mono'; color:#1DB954;
                                 font-size:1.15rem; margin-left:16px;">
                        {row[acc_col]:.4f}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── KPI summary row ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if acc_col:
        k1, k2, k3, k4 = st.columns(4)
        best_idx  = model_df[acc_col].idxmax()
        worst_idx = model_df[acc_col].idxmin()
        for col_w, icon, val, lbl in [
            (k1, "🏆", f"{model_df.loc[best_idx, acc_col]:.4f}",  f"Best · {model_df.loc[best_idx, model_col]}"),
            (k2, "📊", f"{model_df[acc_col].mean():.4f}",          "Mean Accuracy"),
            (k3, "📉", f"{model_df.loc[worst_idx, acc_col]:.4f}", f"Lowest · {model_df.loc[worst_idx, model_col]}"),
            (k4, "🔢", str(len(model_df)),                          "Models Evaluated"),
        ]:
            with col_w:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div style="font-size:1.5rem; margin-bottom:6px;">{icon}</div>
                        <div class="metric-value" style="font-size:1.6rem;">{val}</div>
                        <div class="metric-label">{lbl}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Accuracy bar ───────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">📊 Accuracy Bar Chart</div>',
        unsafe_allow_html=True,
    )
    if acc_col:
        fig_bar = model_accuracy_bar(model_df, acc_col, model_col)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Multi-metric grouped bar ───────────────────────────────────────────────
    metric_candidates = [c for c in model_df.columns
                          if c != model_col
                          and model_df[c].dtype in [np.float64, np.float32, float]]
    if len(metric_candidates) > 1:
        st.markdown(
            '<div class="section-header">📐 Multi-Metric Comparison</div>',
            unsafe_allow_html=True,
        )
        fig_multi = multi_metric_bar(model_df, model_col, metric_candidates[:6])
        st.plotly_chart(fig_multi, use_container_width=True)

    # ── Full table ─────────────────────────────────────────────────────────────
    with st.expander("📋 Full Classification Results Table"):
        st.dataframe(
            model_df.style.background_gradient(cmap="RdYlGn", axis=0),
            use_container_width=True,
        )

else:
    st.info("ℹ️  `models/model_results.csv` not found — train models first.")

# ── Cross-Validation ───────────────────────────────────────────────────────────
if cv_data:
    st.markdown(
        '<div class="section-header">🔄 Cross-Validation Results</div>',
        unsafe_allow_html=True,
    )

    # Build a summary table from the nested-dict structure
    cv_rows = []
    for model_name, raw in cv_data.items():
        if isinstance(raw, dict):
            scores_raw = raw.get("scores", [])
            mean_val   = raw.get("mean_auc")
            std_val    = raw.get("std_auc")
        elif isinstance(raw, list):
            scores_raw = raw
            mean_val, std_val = None, None
        else:
            continue

        valid = [float(s) for s in scores_raw if s is not None and s == s]
        if not valid:
            continue

        mean_v = mean_val if (isinstance(mean_val, float) and mean_val == mean_val) else float(np.mean(valid))
        std_v  = std_val  if (isinstance(std_val,  float) and std_val  == std_val)  else float(np.std(valid))
        cv_rows.append({
            "Model":     model_name.replace("_", " ").title(),
            "Mean AUC":  round(mean_v, 6),
            "Std AUC":   round(std_v, 6),
            "CV Folds":  len(valid),
        })

    if cv_rows:
        cv_summary_df = pd.DataFrame(cv_rows).sort_values("Mean AUC", ascending=False).reset_index(drop=True)
        st.dataframe(
            cv_summary_df.style.background_gradient(subset=["Mean AUC"], cmap="RdYlGn"),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

    # Boxplot (now handles nested dicts internally)
    fig_cv = cv_boxplot(cv_data)
    st.plotly_chart(fig_cv, use_container_width=True)

elif not cv_data:
    st.info("ℹ️  `models/cv_results.json` not found.")

# ── Optimal Thresholds ─────────────────────────────────────────────────────────
if thresholds:
    st.markdown(
        '<div class="section-header">🎯 Optimal Classification Thresholds</div>',
        unsafe_allow_html=True,
    )
    th_cols = st.columns(min(4, len(thresholds)))
    for i, (model_name, thresh_val) in enumerate(thresholds.items()):
        with th_cols[i % len(th_cols)]:
            val_str = f"{thresh_val:.3f}" if isinstance(thresh_val, float) else str(thresh_val)
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:1.4rem; margin-bottom:4px;">🎚️</div>
                    <div class="metric-value" style="font-size:1.5rem;">{val_str}</div>
                    <div class="metric-label">{model_name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ── Regression Results ─────────────────────────────────────────────────────────
if not reg_df.empty:
    st.markdown(
        '<div class="section-header">📈 Regression Model Results</div>',
        unsafe_allow_html=True,
    )

    reg_model_col = reg_df.columns[0]
    r2_col = detect_column(reg_df, "r2")
    if r2_col:
        fig_reg = regression_bar(reg_df, reg_model_col, r2_col)
        st.plotly_chart(fig_reg, use_container_width=True)

    st.dataframe(
        reg_df.style.background_gradient(cmap="RdYlGn", axis=0),
        use_container_width=True,
    )

# ── Static images ──────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">🖼️ Comparison Charts</div>',
    unsafe_allow_html=True,
)

img_pairs = [
    ("model_comparison.png",    "Model Comparison"),
    ("regression_comparison.png","Regression Comparison"),
    ("cv_results.png",          "CV Results"),
    ("threshold_optimization.png","Threshold Optimisation"),
]

sc1, sc2 = st.columns(2)
rendered = 0
for fname, title in img_pairs:
    path = get_viz_path(fname)
    if os.path.exists(path):
        with [sc1, sc2][rendered % 2]:
            st.markdown(f"**{title}**")
            st.image(path, use_column_width=True)
        rendered += 1

if rendered == 0:
    st.info("No static comparison images found in `visualizations/`.")
