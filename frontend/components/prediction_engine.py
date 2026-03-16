"""
Prediction logic for the virality prediction pipeline.

The scaler + stacking ensemble were trained on 61 engineered features.
This module reconstructs the full feature vector from the 9 raw audio
inputs that the user provides via sliders.

Feature engineering mirrors the training notebook:
  raw inputs → derived numeric → polynomial → one-hot buckets
"""

import numpy as np
import pandas as pd
from components.model_loader import load_scaler, load_ensemble

# ── Raw user-facing features ──────────────────────────────────────────────────
FEATURE_COLUMNS = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence',
    'tempo', 'duration_ms', 'key', 'mode', 'explicit',
    'time_signature',
]

# Human-readable UI labels
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

# Default preset — matches the mean audio profile of viral songs in the training set
# (low valence ~0.29, high speechiness ~0.16, loud ~-7.3 dB)
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

# Full ordered feature list exactly as trained (61 features)
ALL_FEATURES = [
    'duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'time_signature', 'duration_min', 'optimal_length', 'energy_valence',
    'danceability_energy', 'vocal_instrumental_balance', 'acoustic_energy_balance',
    'is_live', 'highly_danceable', 'dance_tempo', 'tempo_normalized',
    'loudness_normalized', 'is_loud', 'party_index', 'chill_index', 'workout_index',
    'release_year', 'release_month', 'release_day_of_week', 'is_weekend_release',
    'song_age_years', 'artist_avg_popularity', 'artist_song_count',
    'artist_max_popularity', 'artist_std_popularity', 'artist_has_viral_hit',
    'key_target_enc', 'mode_target_enc', 'time_signature_target_enc',
    'poly_danceability_x_energy', 'poly_danceability_x_valence',
    'poly_danceability_x_loudness_normalized', 'poly_energy_x_valence',
    'poly_energy_x_loudness_normalized', 'poly_valence_x_loudness_normalized',
    'mood_happy_energetic', 'mood_neutral', 'mood_sad_calm', 'mood_sad_energetic',
    'duration_category_short', 'duration_category_medium', 'duration_category_long',
    'tempo_category_moderate', 'tempo_category_fast', 'tempo_category_very_fast',
    'popularity_bucket_medium', 'popularity_bucket_high', 'popularity_bucket_viral',
]


def _build_feature_vector(raw: dict) -> pd.DataFrame:
    """
    Build the complete 61-feature DataFrame from raw user inputs.
    Mirrors the feature engineering done during model training.
    """
    d  = raw.copy()
    f  = {}

    # ── Passthrough raw features ──────────────────────────────────────────────
    for col in ['duration_ms', 'explicit', 'danceability', 'energy', 'key',
                'loudness', 'mode', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']:
        f[col] = float(d.get(col, 0.0))

    dur_ms     = f['duration_ms']
    energy     = f['energy']
    valence    = f['valence']
    dance      = f['danceability']
    loudness   = f['loudness']
    tempo      = f['tempo']
    instr      = f['instrumentalness']
    acoustic   = f['acousticness']
    liveness   = f['liveness']

    # ── Derived numeric features ──────────────────────────────────────────────
    f['duration_min']               = dur_ms / 60000.0
    # optimal_length: 1 if 2.5–4 min, else 0
    f['optimal_length']             = 1.0 if 150000 <= dur_ms <= 240000 else 0.0
    f['energy_valence']             = energy * valence
    f['danceability_energy']        = dance * energy
    f['vocal_instrumental_balance'] = 1.0 - instr
    f['acoustic_energy_balance']    = acoustic * (1.0 - energy)
    f['is_live']                    = 1.0 if liveness > 0.8 else 0.0
    f['highly_danceable']           = 1.0 if dance > 0.7 else 0.0
    f['dance_tempo']                = dance * tempo

    # Normalise tempo to ~[0,1] assuming 60–180 BPM range
    f['tempo_normalized']           = max(0.0, min(1.0, (tempo - 60.0) / 120.0))
    # Normalise loudness: -60→0, 0→1
    f['loudness_normalized']        = max(0.0, min(1.0, (loudness + 60.0) / 60.0))
    f['is_loud']                    = 1.0 if loudness > -8.0 else 0.0

    # Composite indices
    f['party_index']                = (dance + energy + valence) / 3.0
    f['chill_index']                = (acoustic + (1.0 - energy) + (1.0 - dance)) / 3.0
    f['workout_index']              = (energy + f['tempo_normalized'] + dance) / 3.0

    # Date-related features — use sensible defaults for a "new" track
    import datetime
    today = datetime.date.today()
    f['release_year']               = float(today.year)
    f['release_month']              = float(today.month)
    f['release_day_of_week']        = float(today.weekday())   # 0=Mon
    f['is_weekend_release']         = 1.0 if today.weekday() >= 5 else 0.0
    f['song_age_years']             = 0.0   # brand-new track

    # Artist-level features — use above-average defaults for an established artist
    f['artist_avg_popularity']      = 65.0
    f['artist_song_count']          = 12.0
    f['artist_max_popularity']      = 78.0
    f['artist_std_popularity']      = 12.0
    f['artist_has_viral_hit']       = 1.0

    # Target-encoded categoricals — use overall mean (0.5 is safe neutral)
    f['key_target_enc']             = 0.5
    f['mode_target_enc']            = 0.55
    f['time_signature_target_enc']  = 0.5

    # ── Polynomial interaction features ───────────────────────────────────────
    ln = f['loudness_normalized']
    f['poly_danceability_x_energy']              = dance * energy
    f['poly_danceability_x_valence']             = dance * valence
    f['poly_danceability_x_loudness_normalized'] = dance * ln
    f['poly_energy_x_valence']                   = energy * valence
    f['poly_energy_x_loudness_normalized']        = energy * ln
    f['poly_valence_x_loudness_normalized']       = valence * ln

    # ── Mood buckets (one-hot, exactly 4 categories) ──────────────────────────
    happy_energetic = energy >= 0.6 and valence >= 0.6
    sad_calm        = energy <  0.4 and valence <  0.4
    sad_energetic   = energy >= 0.6 and valence <  0.4
    neutral         = not (happy_energetic or sad_calm or sad_energetic)

    f['mood_happy_energetic'] = 1.0 if happy_energetic else 0.0
    f['mood_neutral']         = 1.0 if neutral         else 0.0
    f['mood_sad_calm']        = 1.0 if sad_calm        else 0.0
    f['mood_sad_energetic']   = 1.0 if sad_energetic   else 0.0

    # ── Duration bucket (one-hot: short < 3min, medium 3-5min, long > 5min) ──
    short  = dur_ms < 180000
    long_  = dur_ms > 300000
    medium = not short and not long_
    f['duration_category_short']  = 1.0 if short  else 0.0
    f['duration_category_medium'] = 1.0 if medium else 0.0
    f['duration_category_long']   = 1.0 if long_  else 0.0

    # ── Tempo bucket (one-hot: slow <80, moderate 80-110, fast 110-140, very_fast >140) ──
    moderate   = 80 <= tempo < 110
    fast       = 110 <= tempo < 140
    very_fast  = tempo >= 140
    f['tempo_category_moderate']  = 1.0 if moderate  else 0.0
    f['tempo_category_fast']      = 1.0 if fast       else 0.0
    f['tempo_category_very_fast'] = 1.0 if very_fast  else 0.0

    # ── Popularity bucket (one-hot: low, medium, high, viral) ────────────────
    # Popularity is unknown at prediction time.
    # Use 'high' so predictions reflect an established track getting attention.
    f['popularity_bucket_medium'] = 0.0
    f['popularity_bucket_high']   = 1.0
    f['popularity_bucket_viral']  = 0.0

    # Build ordered array matching ALL_FEATURES
    row = [f[feat] for feat in ALL_FEATURES]
    return pd.DataFrame([row], columns=ALL_FEATURES)


def predict_virality(feature_dict: dict) -> dict:
    """
    Run the full prediction pipeline for one song.

    Parameters
    ----------
    feature_dict : dict  — user-supplied raw audio features

    Returns
    -------
    dict with keys: prediction (int|None), probability (float|None),
                    confidence_label (str), error (bool)
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

    try:
        X = _build_feature_vector(feature_dict)
        X_scaled = scaler.transform(X)

        pred  = int(model.predict(X_scaled)[0])
        proba = model.predict_proba(X_scaled)[0]
        viral_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])

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

    except Exception as e:
        return {
            'prediction':       None,
            'probability':      None,
            'confidence_label': f'Pipeline error: {e}',
            'error':            True,
        }


def get_feature_ranges() -> dict:
    """
    Return (min, max, default, step) slider params for the user-facing features.
    """
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
