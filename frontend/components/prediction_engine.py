"""
Prediction logic for the virality prediction pipeline.

Pipeline:  raw feature dict  →  StandardScaler  →  StackingEnsemble  →  result dict
"""

import numpy as np
from components.model_loader import load_scaler, load_ensemble

# ── Feature contract ──────────────────────────────────────────────────────────
# Must match the column order used during training.
FEATURE_COLUMNS = [
    'danceability',
    'energy',
    'loudness',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'valence',
    'tempo',
    'duration_ms',
]

# Human-readable labels for the UI
FEATURE_LABELS = {
    'danceability':      '💃 Danceability',
    'energy':            '⚡ Energy',
    'loudness':          '🔊 Loudness (dB)',
    'speechiness':       '🗣️ Speechiness',
    'acousticness':      '🎸 Acousticness',
    'instrumentalness':  '🎹 Instrumentalness',
    'valence':           '😊 Valence',
    'tempo':             '🥁 Tempo (BPM)',
    'duration_ms':       '⏱️ Duration (ms)',
}

# Default "average-pop song" preset
DEFAULT_FEATURES = {
    'danceability':     0.65,
    'energy':           0.70,
    'loudness':         -6.0,
    'speechiness':      0.05,
    'acousticness':     0.20,
    'instrumentalness': 0.01,
    'valence':          0.50,
    'tempo':            120.0,
    'duration_ms':      200000,
}


def predict_virality(feature_dict: dict) -> dict:
    """
    Run the full prediction pipeline for one song.

    Parameters
    ----------
    feature_dict : dict mapping FEATURE_COLUMNS keys → float values

    Returns
    -------
    dict with keys:
        prediction        (int | None)  – 1 = viral, 0 = not viral
        probability       (float | None) – P(viral)
        confidence_label  (str)
        error             (bool)
    """
    scaler = load_scaler()
    model  = load_ensemble()

    if scaler is None or model is None:
        return {
            'prediction':       None,
            'probability':      None,
            'confidence_label': 'Model unavailable — check models/ directory',
            'error':            True,
        }

    # Build feature vector in the exact training column order
    features = [float(feature_dict.get(col, 0.0)) for col in FEATURE_COLUMNS]
    X = np.array(features, dtype=float).reshape(1, -1)

    # Scale
    X_scaled = scaler.transform(X)

    # Predict
    pred  = int(model.predict(X_scaled)[0])
    proba = model.predict_proba(X_scaled)[0]

    # Probability of class "1" (viral)
    viral_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])

    # Human-readable confidence tier
    if viral_prob >= 0.80:
        confidence = "Very High Confidence"
    elif viral_prob >= 0.65:
        confidence = "High Confidence"
    elif viral_prob >= 0.50:
        confidence = "Moderate Confidence"
    elif viral_prob >= 0.35:
        confidence = "Below Average"
    else:
        confidence = "Low Probability"

    return {
        'prediction':       pred,
        'probability':      viral_prob,
        'confidence_label': confidence,
        'error':            False,
    }


def get_feature_ranges() -> dict:
    """
    Return (min, max, default, step) slider params for every feature.
    Used by the Predict page to build sliders dynamically.
    """
    return {
        'danceability':     (0.0,   1.0,    0.65,   0.01),
        'energy':           (0.0,   1.0,    0.70,   0.01),
        'loudness':         (-60.0, 0.0,   -6.0,    0.5),
        'speechiness':      (0.0,   1.0,    0.05,   0.01),
        'acousticness':     (0.0,   1.0,    0.20,   0.01),
        'instrumentalness': (0.0,   1.0,    0.01,   0.001),
        'valence':          (0.0,   1.0,    0.50,   0.01),
        'tempo':            (50.0,  220.0,  120.0,  1.0),
        'duration_ms':      (30000, 600000, 200000, 1000),
    }
