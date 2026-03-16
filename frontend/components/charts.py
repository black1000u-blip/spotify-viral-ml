"""
Reusable Plotly chart builders — all rendered with the dashboard dark theme.

Every function returns a go.Figure ready to pass to st.plotly_chart().
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ── Design tokens ─────────────────────────────────────────────────────────────
GREEN  = "#1DB954"
CYAN   = "#00d4ff"
PURPLE = "#9b59b6"
RED    = "#e74c3c"
GOLD   = "#f39c12"

COLOR_SEQ = [GREEN, CYAN, PURPLE, GOLD, RED, "#2ecc71", "#3498db"]

# Shared layout overrides applied to every chart
_BASE_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,17,24,0.6)",
    font=dict(family="JetBrains Mono, monospace", color="#f0f0f5"),
    margin=dict(l=20, r=20, t=48, b=20),
)


def _rgb(hex_color: str, alpha: float = 1.0) -> str:
    """Convert #rrggbb to rgba(r,g,b,a) string."""
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ── Chart builders ─────────────────────────────────────────────────────────────

def model_accuracy_bar(df: pd.DataFrame,
                        metric_col: str = "accuracy",
                        model_col: str = "model") -> go.Figure:
    """Horizontal bar chart — model accuracy leaderboard."""
    df_sorted = df.sort_values(metric_col, ascending=True).copy()
    max_val = df_sorted[metric_col].max()
    colors = [GREEN if v == max_val else CYAN for v in df_sorted[metric_col]]

    fig = go.Figure(go.Bar(
        x=df_sorted[metric_col],
        y=df_sorted[model_col],
        orientation='h',
        marker=dict(color=colors, line=dict(width=0), opacity=0.85),
        text=[f"{v:.4f}" for v in df_sorted[metric_col]],
        textposition='outside',
        textfont=dict(family="JetBrains Mono", size=11),
    ))
    fig.update_layout(
        title="Model Accuracy Comparison",
        xaxis_title="Accuracy",
        xaxis=dict(range=[df_sorted[metric_col].min() * 0.97, 1.01]),
        **_BASE_LAYOUT,
    )
    return fig


def multi_metric_bar(df: pd.DataFrame,
                      model_col: str,
                      metrics: list) -> go.Figure:
    """Grouped bar chart comparing multiple metrics across models."""
    fig = go.Figure()
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            color = COLOR_SEQ[i % len(COLOR_SEQ)]
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=df[model_col],
                y=df[metric],
                marker=dict(color=color, opacity=0.82),
                text=[f"{v:.3f}" for v in df[metric]],
                textposition='outside',
                textfont=dict(size=9),
            ))
    fig.update_layout(
        barmode='group',
        title="Multi-Metric Model Comparison",
        yaxis_title="Score",
        **_BASE_LAYOUT,
    )
    return fig


def cv_boxplot(cv_data: dict) -> go.Figure:
    """Box plot of cross-validation scores per model.

    Handles two JSON layouts:
      • {model: [s1, s2, ...]}                        – plain list
      • {model: {"scores": [...], "mean_auc": x, ...}} – nested dict from pipeline
    """
    fig = go.Figure()
    added = 0

    for i, (model, raw) in enumerate(cv_data.items()):
        color = COLOR_SEQ[i % len(COLOR_SEQ)]

        # ── Normalise to a plain list of floats ───────────────────────────────
        if isinstance(raw, dict):
            scores_raw = raw.get("scores", [])
            mean_lbl   = raw.get("mean_auc")
        elif isinstance(raw, list):
            scores_raw = raw
            mean_lbl   = None
        else:
            continue

        # Filter out NaN values; skip models whose entire score list is NaN
        scores = [float(s) for s in scores_raw if s is not None and s == s]  # NaN != NaN
        if not scores:
            continue

        hover_name = (
            f"{model}<br>Mean: {mean_lbl:.4f}" if isinstance(mean_lbl, float) and mean_lbl == mean_lbl
            else model
        )

        fig.add_trace(go.Box(
            y=scores,
            name=model,
            marker_color=color,
            line_color=color,
            fillcolor=_rgb(color, 0.15),
            boxmean=True,
            boxpoints="outliers",
            hovertemplate=f"<b>{hover_name}</b><br>Score: %{{y:.4f}}<extra></extra>",
        ))
        added += 1

    if added == 0:
        # Return empty figure with a note
        fig.add_annotation(
            text="No valid CV scores available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#8888aa", size=14),
        )

    fig.update_layout(
        title="Cross-Validation AUC Score Distribution",
        yaxis_title="AUC Score",
        showlegend=False,
        **_BASE_LAYOUT,
    )
    return fig


def scatter_virality(df: pd.DataFrame,
                      x_col: str,
                      y_col: str,
                      color_col: str = "viral") -> go.Figure:
    """Scatter plot of two audio features coloured by virality label."""
    if x_col not in df.columns or y_col not in df.columns:
        return go.Figure()

    if color_col in df.columns:
        fig = px.scatter(
            df, x=x_col, y=y_col,
            color=color_col.astype(str) if hasattr(color_col, 'astype') else color_col,
            color_discrete_map={
                "0": RED, "1": GREEN,
                0: RED, 1: GREEN,
                False: RED, True: GREEN,
            },
            opacity=0.60,
            title=f"{x_col.replace('_',' ').title()} vs {y_col.replace('_',' ').title()}",
        )
    else:
        fig = px.scatter(df, x=x_col, y=y_col, opacity=0.60,
                         title=f"{x_col} vs {y_col}",
                         color_discrete_sequence=[CYAN])

    fig.update_traces(marker=dict(size=4))
    fig.update_layout(**_BASE_LAYOUT)
    return fig


def tempo_histogram(df: pd.DataFrame, color_col: str = "viral") -> go.Figure:
    """Overlapping histogram of tempo split by viral / non-viral."""
    fig = go.Figure()
    if color_col in df.columns and "tempo" in df.columns:
        for val, color, label in [(1, GREEN, "Viral"), (0, RED, "Non-Viral")]:
            subset = df[df[color_col] == val]["tempo"].dropna()
            fig.add_trace(go.Histogram(
                x=subset, name=label,
                marker_color=color, opacity=0.65, nbinsx=40,
            ))
        fig.update_layout(barmode="overlay")
    elif "tempo" in df.columns:
        fig.add_trace(go.Histogram(
            x=df["tempo"].dropna(), marker_color=CYAN, nbinsx=40,
        ))

    fig.update_layout(
        title="Tempo Distribution — Viral vs Non-Viral",
        xaxis_title="Tempo (BPM)",
        yaxis_title="Count",
        **_BASE_LAYOUT,
    )
    return fig


def correlation_heatmap(df: pd.DataFrame, cols: list = None) -> go.Figure:
    """Interactive annotated correlation heatmap."""
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()[:12]
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return go.Figure()

    corr = df[cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[[0, RED], [0.5, "#1a1a2e"], [1, GREEN]],
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        showscale=True,
    ))
    fig.update_layout(
        title="Feature Correlation Matrix",
        **_BASE_LAYOUT,
    )
    return fig


def gauge_chart(probability: float) -> go.Figure:
    """Semicircular probability gauge for the virality score."""
    color = GREEN if probability >= 0.5 else RED
    pct = round(probability * 100, 1)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number=dict(suffix="%", font=dict(size=44, family="JetBrains Mono", color=color)),
        delta=dict(reference=50, valueformat=".1f",
                   increasing=dict(color=GREEN), decreasing=dict(color=RED)),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(size=10)),
            bar=dict(color=color, thickness=0.28),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0,  40], color=_rgb(RED,    0.12)),
                dict(range=[40, 60], color=_rgb(GOLD,   0.10)),
                dict(range=[60, 100], color=_rgb(GREEN, 0.12)),
            ],
            threshold=dict(
                line=dict(color=CYAN, width=3),
                thickness=0.80,
                value=50,
            ),
        ),
        title=dict(text="Viral Probability", font=dict(size=14, color="#8888aa")),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(
        height=270,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f0f0f5"),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def feature_radar(feature_dict: dict) -> go.Figure:
    """Radar chart of normalised (0-1) audio features."""
    radar_features = [
        'danceability', 'energy', 'speechiness',
        'acousticness', 'instrumentalness', 'valence',
    ]
    values     = [float(feature_dict.get(f, 0)) for f in radar_features]
    values_cls = values + [values[0]]          # close the polygon
    categories = radar_features + [radar_features[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_cls,
        theta=categories,
        fill='toself',
        fillcolor=_rgb(GREEN, 0.15),
        line=dict(color=GREEN, width=2),
        marker=dict(color=GREEN, size=6),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor="rgba(255,255,255,0.08)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f0f0f5", family="JetBrains Mono"),
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=False,
        title=dict(text="Audio Feature Profile",
                   font=dict(size=14, color="#8888aa")),
        height=320,
    )
    return fig


def popularity_distribution(df: pd.DataFrame,
                              pop_col: str = "track_popularity") -> go.Figure:
    """Gradient histogram of song popularity scores."""
    if pop_col not in df.columns:
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            return go.Figure()
        pop_col = num_cols[0]

    data = df[pop_col].dropna()
    fig = go.Figure(go.Histogram(
        x=data,
        nbinsx=50,
        marker=dict(
            color=data,
            colorscale=[[0, RED], [0.4, PURPLE], [1, GREEN]],
            showscale=False,
        ),
        opacity=0.85,
    ))
    fig.update_layout(
        title="Popularity Score Distribution",
        xaxis_title="Popularity",
        yaxis_title="Count",
        **_BASE_LAYOUT,
    )
    return fig


def regression_bar(df: pd.DataFrame,
                    model_col: str,
                    metric_col: str = "r2") -> go.Figure:
    """Bar chart for regression model comparison."""
    df_s = df.sort_values(metric_col, ascending=False)
    colors = [GREEN if i == 0 else PURPLE for i in range(len(df_s))]
    fig = go.Figure(go.Bar(
        x=df_s[model_col],
        y=df_s[metric_col],
        marker=dict(color=colors, opacity=0.82),
        text=[f"{v:.4f}" for v in df_s[metric_col]],
        textposition='outside',
        textfont=dict(size=10),
    ))
    fig.update_layout(
        title=f"Regression — {metric_col.upper()} Comparison",
        yaxis_title=metric_col.upper(),
        **_BASE_LAYOUT,
    )
    return fig
