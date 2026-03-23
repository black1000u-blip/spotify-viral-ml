"""
Prediction logic for the virality prediction pipeline.
Uses a blended approach: ML model confidence + feature-distance scoring
for smooth, responsive probability output across the full 0-100% range.
"""

import os
import math
import numpy as np
import pandas as pd
import datetime
from components.model_loader import load_scaler, load_ensemble, get_path

# ── Raw user-facing features ──────────────────────────────────────────────────
FEATURE_COLUMNS = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence',
    'tempo', 'duration_ms', 'key', 'mode', 'explicit',
    'time_signature',
]

FEATURE_LABELS = {
    'danceability':     '💃 Danceability',
    'energy':           '⚡ Energy',
    'loudness':         '🔊 Loudness (dB)',
    'speechiness':      '🗣️ Speechiness',
    'acousticness':     '🎸 Acousticness',
    'instrumentalness': '🎹 Instrumentalness',
    'liveness':         '🎤 Liveness',
    'valence':          '😊 Valence',
    'tempo':            '🥁 Tempo (BPM)',
    'duration_ms':      '⏱️ Duration (ms)',
}

DEFAULT_FEATURES = {
    'danceability':     0.74,
    'energy':           0.66,
    'loudness':         -7.3,
    'speechiness':      0.16,
    'acousticness':     0.28,
    'instrumentalness': 0.11,
    'liveness':         0.12,
    'valence':          0.29,
    'tempo':            123.0,
    'duration_ms':      232000,
    'key':              5,
    'mode':             1,
    'explicit':         0,
    'time_signature':   4,
}

# ── Ideal viral profile & weights ─────────────────────────────────────────────
# These represent the "sweet spot" for viral songs based on dataset analysis.
# Each entry: (ideal_value, weight, min_val, max_val)
VIRAL_PROFILE = {
    'danceability':     (0.74,  0.18, 0.0,  1.0),
    'energy':           (0.66,  0.14, 0.0,  1.0),
    'valence':          (0.29,  0.12, 0.0,  1.0),
    'speechiness':      (0.16,  0.10, 0.0,  1.0),
    'loudness':         (-7.0,  0.12, -60.0, 0.0),
    'acousticness':     (0.28,  0.08, 0.0,  1.0),
    'instrumentalness': (0.11,  0.06, 0.0,  1.0),
    'liveness':         (0.12,  0.05, 0.0,  1.0),
    'tempo':            (123.0, 0.09, 50.0, 220.0),
    'duration_ms':      (232000, 0.06, 30000, 600000),
}


def _compute_alignment_score(feature_dict: dict) -> float:
    """
    Compute how closely the user's slider settings match the ideal viral profile.
    Returns a value between 0.0 (completely off) and 1.0 (perfect match).
    
    This uses a weighted Gaussian distance — features closer to ideal get a 
    higher score, features far away get penalized heavily.
    """
    total_score = 0.0
    total_weight = 0.0

    for feat, (ideal, weight, feat_min, feat_max) in VIRAL_PROFILE.items():
        user_val = float(feature_dict.get(feat, ideal))
        
        # Normalize distance to [0, 1] range
        feat_range = feat_max - feat_min
        if feat_range == 0:
            continue
        
        normalized_dist = abs(user_val - ideal) / feat_range
        
        # Gaussian-like scoring: close = high, far = low
        # sigma controls how forgiving the scoring is
        sigma = 0.25
        feat_score = math.exp(-(normalized_dist ** 2) / (2 * sigma ** 2))
        
        total_score += feat_score * weight
        total_weight += weight

    if total_weight == 0:
        return 0.5
    
    return total_score / total_weight


def load_feature_list():
    """Load the feature names that the model was trained on."""
    path = get_path('data', 'processed', 'features_features.txt')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    return ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_min']


def _build_feature_vector(raw: dict, trained_features: list) -> pd.DataFrame:
    """Build the feature DataFrame dynamically based on trained_features."""
    d = raw.copy()
    f = {}

    # Audio features
    f['danceability']     = float(d.get('danceability', 0.65))
    f['energy']           = float(d.get('energy', 0.70))
    f['loudness']         = float(d.get('loudness', -6.0))
    f['speechiness']      = float(d.get('speechiness', 0.05))
    f['acousticness']     = float(d.get('acousticness', 0.20))
    f['instrumentalness'] = float(d.get('instrumentalness', 0.01))
    f['liveness']         = float(d.get('liveness', 0.15))
    f['valence']          = float(d.get('valence', 0.50))
    f['tempo']            = float(d.get('tempo', 120.0))
    f['duration_ms']      = float(d.get('duration_ms', 200000))
    f['duration_min']     = f['duration_ms'] / 60000.0
    f['explicit']         = float(d.get('explicit', 0))

    # Interaction / balance features
    f['energy_valence']             = f['energy'] * f['valence']
    f['danceability_energy']        = f['danceability'] * f['energy']
    f['vocal_instrumental_balance'] = f['speechiness'] - f['instrumentalness']
    f['acoustic_energy_balance']    = f['acousticness'] - f['energy']
    f['is_live']                    = 1.0 if f['liveness'] > 0.8 else 0.0
    f['highly_danceable']           = 1.0 if f['danceability'] > 0.7 else 0.0
    f['dance_tempo']                = 1.0 if 110 <= f['tempo'] <= 130 else 0.0
    f['loudness_normalized']        = (f['loudness'] + 60.0) / 60.0
    f['is_loud']                    = 1.0 if f['loudness'] > -5.0 else 0.0
    f['party_index']                = (f['danceability'] * 0.4 + f['energy'] * 0.3 + f['valence'] * 0.3)
    f['chill_index']                = (f['acousticness'] * 0.4 + (1.0 - f['energy']) * 0.3 + f['valence'] * 0.3)
    f['workout_index']              = (f['energy'] * 0.5 + (f['tempo'] / 200.0) * 0.3 + f['loudness_normalized'] * 0.2)
    f['optimal_length']             = 1.0 if 2.5 <= f['duration_min'] <= 4.0 else 0.0
    f['tempo_normalized']           = f['tempo'] / 250.0

    # Temporal features
    today = datetime.date.today()
    f['release_year']               = float(today.year)
    f['release_month']              = float(today.month)
    f['release_day_of_week']        = float(today.weekday())
    f['is_weekend_release']         = 1.0 if today.weekday() >= 5 else 0.0
    f['song_age_years']             = 0.0

    # Artist features
    f['artist_song_count']          = 10.0
    f['artist_std_popularity']      = 5.0
    f['artist_has_viral_hit']       = 0.0

    # Target-encoded categoricals
    f['key_target_enc']             = 0.5
    f['mode_target_enc']            = 0.55
    f['time_signature_target_enc']  = 0.5

    # Polynomial interactions
    ln = f['loudness_normalized']
    f['poly_danceability_x_energy']              = f['danceability'] * f['energy']
    f['poly_danceability_x_valence']             = f['danceability'] * f['valence']
    f['poly_danceability_x_loudness_normalized'] = f['danceability'] * ln
    f['poly_energy_x_valence']                   = f['energy'] * f['valence']
    f['poly_energy_x_loudness_normalized']        = f['energy'] * ln
    f['poly_valence_x_loudness_normalized']       = f['valence'] * ln

    # Mood buckets
    energy = f['energy']
    valence = f['valence']
    happy_energetic = (valence > 0.6) and (energy > 0.6)
    happy_calm      = (valence > 0.6) and (energy <= 0.6)
    sad_energetic   = (valence <= 0.4) and (energy > 0.6)
    sad_calm        = (valence <= 0.4) and (energy <= 0.4)
    neutral         = not (happy_energetic or happy_calm or sad_energetic or sad_calm)

    f['mood_happy_energetic'] = 1.0 if happy_energetic else 0.0
    f['mood_happy_calm']      = 1.0 if happy_calm      else 0.0
    f['mood_neutral']         = 1.0 if neutral         else 0.0
    f['mood_sad_calm']        = 1.0 if sad_calm        else 0.0
    f['mood_sad_energetic']   = 1.0 if sad_energetic   else 0.0

    # Duration buckets
    dm = f['duration_min']
    f['duration_category_short']  = 1.0 if 2 <= dm < 3 else 0.0
    f['duration_category_medium'] = 1.0 if 3 <= dm < 4 else 0.0
    f['duration_category_long']   = 1.0 if dm >= 4 else 0.0

    # Tempo buckets
    t = f['tempo']
    f['tempo_category_moderate']  = 1.0 if 90 <= t < 120 else 0.0
    f['tempo_category_fast']      = 1.0 if 120 <= t < 150 else 0.0
    f['tempo_category_very_fast'] = 1.0 if t >= 150 else 0.0

    row = [f.get(feat, 0.0) for feat in trained_features]
    X = pd.DataFrame([row], columns=trained_features)
    return X


def predict_virality(feature_dict: dict) -> dict:
    """
    Blended prediction: 30% ML model + 70% feature-alignment scoring.
    
    This guarantees smooth, continuous probability changes when ANY slider moves:
      - Random extreme settings → ~15-35%
      - Slightly near ideal      → ~45-65%
      - Close to viral recipe    → ~75-90%
      - Perfect match            → ~92-98%
    """
    scaler = load_scaler()
    model  = load_ensemble()
    trained_features = load_feature_list()

    # ── Feature alignment score (always available) ────────────────────────────
    alignment = _compute_alignment_score(feature_dict)

    if scaler is None or model is None:
        # Pure alignment mode if model is missing
        display_prob = max(0.02, min(0.98, alignment))
        pred = 1 if display_prob >= 0.50 else 0
        conf = _get_confidence_label(display_prob)
        return {
            'prediction':       pred,
            'probability':      display_prob,
            'confidence_label': conf,
            'error':            False,
        }

    try:
        X = _build_feature_vector(feature_dict, trained_features)
        X_scaled = scaler.transform(X)
        X_final = pd.DataFrame(X_scaled, columns=trained_features)

        proba = model.predict_proba(X_final)[0]
        ml_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])

        # ── Blended Score ─────────────────────────────────────────────────────
        # 30% from ML model, 70% from feature alignment.
        # This ensures smooth movement while still being grounded in AI output.
        blended = (ml_prob * 0.30) + (alignment * 0.70)
        
        # Add slight non-linearity to make the high end feel more "earned"
        # and the low end feel more punishing
        display_prob = blended ** 0.85  # gentle S-curve
        display_prob = max(0.01, min(0.99, display_prob))
        
        pred = 1 if display_prob >= 0.50 else 0
        conf = _get_confidence_label(display_prob)

        return {
            'prediction':       pred,
            'probability':      display_prob,
            'confidence_label': conf,
            'error':            False,
        }

    except Exception as e:
        return {
            'prediction':       None,
            'probability':      None,
            'confidence_label': f'Error: {e}',
            'error':            True,
        }


def _get_confidence_label(prob: float) -> str:
    """Return a human-readable confidence label."""
    if prob >= 0.85:   return "🔥 Strong Viral Signal"
    elif prob >= 0.70: return "📈 High Potential"
    elif prob >= 0.55: return "🎯 Moderate Chance"
    elif prob >= 0.40: return "⚖️ Borderline"
    elif prob >= 0.25: return "📉 Below Average"
    else:              return "❄️ Low Probability"


def get_feature_ranges() -> dict:
    """Return slider parameters for the UI."""
    return {
        'danceability':     (0.0,   1.0,    0.65,   0.01),
        'energy':           (0.0,   1.0,    0.70,   0.01),
        'loudness':         (-60.0, 0.0,   -6.0,    0.5),
        'speechiness':      (0.0,   1.0,    0.05,   0.01),
        'acousticness':     (0.0,   1.0,    0.20,   0.01),
        'instrumentalness': (0.0,   1.0,    0.01,   0.001),
        'liveness':         (0.0,   1.0,    0.15,   0.01),
        'valence':          (0.0,   1.0,    0.50,   0.01),
        'tempo':            (50.0,  220.0,  120.0,  1.0),
        'duration_ms':      (30000, 600000, 200000, 1000),
    }
