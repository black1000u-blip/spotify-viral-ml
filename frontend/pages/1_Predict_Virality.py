"""
Page 1 — Predict Virality

Flow:
  User adjusts audio-feature sliders
  → clicks "Predict Virality"
  → StackingEnsemble (via StandardScaler) returns probability
  → Result card  +  Gauge chart  +  Radar chart
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── CSS ────────────────────────────────────────────────────────────────────────
_css = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
if os.path.exists(_css):
    with open(_css) as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

from components.prediction_engine import (
    predict_virality,
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    DEFAULT_FEATURES,
    get_feature_ranges,
)
from components.charts import gauge_chart, feature_radar
from components.model_loader import load_scaler, load_ensemble

# ── Page heading ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="page-title">🎤 Predict Virality</div>
    <div class="page-subtitle">
        Tune the audio features below and let the AI Stacking Ensemble decide
        whether your track has what it takes to go viral.
    </div>
    """,
    unsafe_allow_html=True,
)

# Model availability check
scaler_ok = load_scaler() is not None
model_ok  = load_ensemble() is not None

if not (scaler_ok and model_ok):
    st.warning(
        "⚠️  One or more model files could not be found. "
        "Make sure `models/stacking_ensemble.pkl` and `models/scaler_standard.pkl` exist. "
        "You can still explore the sliders, but predictions will be unavailable.",
        icon="⚠️",
    )

# ── Preset selector ────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">🎛️ Quick Presets</div>',
    unsafe_allow_html=True,
)

PRESETS = {
    "🎵 Default (Average Pop)":  DEFAULT_FEATURES.copy(),
    "🔥 Viral Banger":           {
        "danceability": 0.88, "energy": 0.92, "loudness": -3.5,
        "speechiness": 0.07, "acousticness": 0.04, "instrumentalness": 0.00,
        "valence": 0.78, "tempo": 128.0, "duration_ms": 195000,
    },
    "🎸 Indie Chill":            {
        "danceability": 0.42, "energy": 0.35, "loudness": -12.0,
        "speechiness": 0.04, "acousticness": 0.75, "instrumentalness": 0.02,
        "valence": 0.38, "tempo": 95.0, "duration_ms": 240000,
    },
    "🎹 Electronic / EDM":       {
        "danceability": 0.82, "energy": 0.95, "loudness": -4.0,
        "speechiness": 0.05, "acousticness": 0.01, "instrumentalness": 0.65,
        "valence": 0.60, "tempo": 138.0, "duration_ms": 210000,
    },
    "🎤 Hip-Hop":                {
        "danceability": 0.80, "energy": 0.70, "loudness": -5.5,
        "speechiness": 0.25, "acousticness": 0.10, "instrumentalness": 0.00,
        "valence": 0.55, "tempo": 90.0, "duration_ms": 215000,
    },
}

_preset_key = st.selectbox(
    "Load a preset",
    options=list(PRESETS.keys()),
    label_visibility="collapsed",
)
_preset = PRESETS[_preset_key]

# ── Sliders ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">🎛️ Audio Features</div>',
    unsafe_allow_html=True,
)

ranges = get_feature_ranges()
# ranges[feat] = (min, max, default, step)
# We destructure to avoid passing 'value' both positionally and as a keyword.

col1, col2, col3 = st.columns(3)

with col1:
    _mn, _mx, _df, _st = ranges["danceability"]
    danceability = st.slider(
        FEATURE_LABELS["danceability"], _mn, _mx,
        value=float(_preset["danceability"]), step=_st,
        help="How suitable a track is for dancing (0 = least, 1 = most)",
    )
    _mn, _mx, _df, _st = ranges["energy"]
    energy = st.slider(
        FEATURE_LABELS["energy"], _mn, _mx,
        value=float(_preset["energy"]), step=_st,
        help="Intensity and perceived activity level",
    )
    _mn, _mx, _df, _st = ranges["valence"]
    valence = st.slider(
        FEATURE_LABELS["valence"], _mn, _mx,
        value=float(_preset["valence"]), step=_st,
        help="Musical positiveness — 0 = sad/tense, 1 = happy/euphoric",
    )

with col2:
    _mn, _mx, _df, _st = ranges["speechiness"]
    speechiness = st.slider(
        FEATURE_LABELS["speechiness"], _mn, _mx,
        value=float(_preset["speechiness"]), step=_st,
        help="Proportion of spoken words in the track",
    )
    _mn, _mx, _df, _st = ranges["acousticness"]
    acousticness = st.slider(
        FEATURE_LABELS["acousticness"], _mn, _mx,
        value=float(_preset["acousticness"]), step=_st,
        help="Confidence the track is acoustic (not electric/synthesised)",
    )
    _mn, _mx, _df, _st = ranges["instrumentalness"]
    instrumentalness = st.slider(
        FEATURE_LABELS["instrumentalness"], _mn, _mx,
        value=float(_preset["instrumentalness"]), step=_st,
        help="Likelihood the track contains no vocals",
    )

with col3:
    _mn, _mx, _df, _st = ranges["loudness"]
    loudness = st.slider(
        FEATURE_LABELS["loudness"], _mn, _mx,
        value=float(_preset["loudness"]), step=_st,
        help="Overall loudness in dB (commercial music: roughly −8 to −3 dB)",
    )
    _mn, _mx, _df, _st = ranges["tempo"]
    tempo = st.slider(
        FEATURE_LABELS["tempo"], _mn, _mx,
        value=float(_preset["tempo"]), step=_st,
        help="Estimated beats per minute",
    )
    _mn, _mx, _df, _st = ranges["duration_ms"]
    duration_ms = st.slider(
        FEATURE_LABELS["duration_ms"], _mn, _mx,
        value=int(_preset["duration_ms"]), step=_st,
        help="Track length in milliseconds (3 min ≈ 180 000 ms)",
    )

feature_dict = {
    "danceability":     danceability,
    "energy":           energy,
    "loudness":         loudness,
    "speechiness":      speechiness,
    "acousticness":     acousticness,
    "instrumentalness": instrumentalness,
    "valence":          valence,
    "tempo":            tempo,
    "duration_ms":      duration_ms,
}

# ── Predict button ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
btn_col, _ = st.columns([1, 3])
with btn_col:
    predict_btn = st.button("🚀  Predict Virality")

# ── Result ─────────────────────────────────────────────────────────────────────
if predict_btn:
    if not (scaler_ok and model_ok):
        st.error(
            "Cannot run prediction: model files are missing. "
            "Please train the models first and place the `.pkl` files in `models/`."
        )
    else:
        with st.spinner("Running prediction pipeline…"):
            result = predict_virality(feature_dict)

        if result.get("error"):
            st.error(
                f"Prediction failed: {result.get('confidence_label', 'Unknown error')}. "
                "Check that `stacking_ensemble.pkl` and `scaler_standard.pkl` are valid joblib files."
            )
        else:
            prob       = result["probability"]
            pred       = result["prediction"]
            confidence = result["confidence_label"]

            st.markdown(
                '<div class="section-header">🎯 Prediction Result</div>',
                unsafe_allow_html=True,
            )

            res_col, gauge_col, radar_col = st.columns([1.3, 1, 1])

            # ── Result card ────────────────────────────────────────────────────
            with res_col:
                pct_int = int(prob * 100)
                if pred == 1:
                    st.markdown(
                        f"""
                        <div class="result-viral">
                            <div class="result-emoji">🔥</div>
                            <div class="result-label" style="color:#1DB954;">VIRAL HIT</div>
                            <div style="color:#8888aa; font-size:0.85rem; margin-top:8px;">
                                {confidence}
                            </div>
                            <div style="margin-top:22px;">
                                <div class="confidence-bar-container">
                                    <div class="confidence-bar-fill" style="width:{pct_int}%;"></div>
                                </div>
                                <div style="font-family:'JetBrains Mono'; font-size:1.9rem;
                                            color:#1DB954; margin-top:10px;">
                                    {prob*100:.1f}% viral
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-nonviral">
                            <div class="result-emoji">📉</div>
                            <div class="result-label" style="color:#e74c3c;">NOT VIRAL</div>
                            <div style="color:#8888aa; font-size:0.85rem; margin-top:8px;">
                                {confidence}
                            </div>
                            <div style="margin-top:22px;">
                                <div class="confidence-bar-container">
                                    <div style="height:100%; border-radius:50px;
                                                background:#e74c3c; width:{pct_int}%;
                                                transition: width 1s ease;"></div>
                                </div>
                                <div style="font-family:'JetBrains Mono'; font-size:1.9rem;
                                            color:#e74c3c; margin-top:10px;">
                                    {prob*100:.1f}% viral
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Quick feature summary
                st.markdown("<br>", unsafe_allow_html=True)
                tip_lines = []
                if danceability > 0.7:
                    tip_lines.append("✅ High danceability")
                if energy > 0.7:
                    tip_lines.append("✅ High energy")
                if valence > 0.6:
                    tip_lines.append("✅ Positive mood")
                if 110 <= tempo <= 135:
                    tip_lines.append("✅ Ideal tempo range")
                if loudness > -8:
                    tip_lines.append("✅ Loud / punchy mix")
                if tip_lines:
                    st.markdown(
                        "<div style='font-size:0.82rem; color:#8888aa; margin-top:8px;'>"
                        + "<br>".join(tip_lines)
                        + "</div>",
                        unsafe_allow_html=True,
                    )

            # ── Gauge ──────────────────────────────────────────────────────────
            with gauge_col:
                st.plotly_chart(gauge_chart(prob), use_container_width=True)

            # ── Radar ──────────────────────────────────────────────────────────
            with radar_col:
                st.plotly_chart(feature_radar(feature_dict), use_container_width=True)
