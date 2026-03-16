"""
src/dynamic_hybrid_model.py
────────────────────────────
Dynamic Hybrid Ensemble Model for Spotify Virality Prediction.

Selects the best pre-trained model based on the audio characteristics
of the input song, then returns a prediction, probability and the name
of the model that was chosen.

Model selection rules
─────────────────────
  danceability > 0.75 AND energy > 0.70  →  XGBoost
  acousticness  > 0.60                   →  Random Forest
  instrumentalness > 0.50                →  LightGBM
  (default)                              →  Stacking Ensemble

Usage
─────
    from src.dynamic_hybrid_model import DynamicHybridModel

    model = DynamicHybridModel()           # loads all models once
    result = model.predict_dynamic(song_features_dict)
    # → {"prediction": 1, "probability": 0.84, "model_used": "XGBoost"}

    # Or use the standalone convenience function:
    from src.dynamic_hybrid_model import predict_dynamic
    result = predict_dynamic(song_features_dict)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Path helpers ──────────────────────────────────────────────────────────────
_HERE        = Path(__file__).resolve().parent          # …/src/
_PROJECT_ROOT = _HERE.parent                            # …/spotify-virality-prediction/
_MODELS_DIR  = _PROJECT_ROOT / "models"


def _model_path(filename: str) -> Path:
    return _MODELS_DIR / filename


# ── Feature order expected by the scaler / models ─────────────────────────────
# (same 61-feature list the training pipeline produced)
ALL_FEATURES: list[str] = [
    "duration_ms", "explicit", "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "time_signature", "duration_min", "optimal_length", "energy_valence",
    "danceability_energy", "vocal_instrumental_balance", "acoustic_energy_balance",
    "is_live", "highly_danceable", "dance_tempo", "tempo_normalized",
    "loudness_normalized", "is_loud", "party_index", "chill_index", "workout_index",
    "release_year", "release_month", "release_day_of_week", "is_weekend_release",
    "song_age_years", "artist_avg_popularity", "artist_song_count",
    "artist_max_popularity", "artist_std_popularity", "artist_has_viral_hit",
    "key_target_enc", "mode_target_enc", "time_signature_target_enc",
    "poly_danceability_x_energy", "poly_danceability_x_valence",
    "poly_danceability_x_loudness_normalized", "poly_energy_x_valence",
    "poly_energy_x_loudness_normalized", "poly_valence_x_loudness_normalized",
    "mood_happy_energetic", "mood_neutral", "mood_sad_calm", "mood_sad_energetic",
    "duration_category_short", "duration_category_medium", "duration_category_long",
    "tempo_category_moderate", "tempo_category_fast", "tempo_category_very_fast",
    "popularity_bucket_medium", "popularity_bucket_high", "popularity_bucket_viral",
]


# ── Feature engineering (mirrors training pipeline) ───────────────────────────

def _build_feature_vector(raw: Dict[str, Any]) -> pd.DataFrame:
    """
    Reconstruct the full 61-feature DataFrame from raw audio inputs.
    Unknown keys default to sensible neutral values.
    """
    import datetime

    f: Dict[str, float] = {}

    # Passthrough raw features
    for col in ["duration_ms", "explicit", "danceability", "energy", "key",
                "loudness", "mode", "speechiness", "acousticness",
                "instrumentalness", "liveness", "valence", "tempo", "time_signature"]:
        f[col] = float(raw.get(col, 0.0))

    dur_ms   = f["duration_ms"]
    energy   = f["energy"]
    valence  = f["valence"]
    dance    = f["danceability"]
    loudness = f["loudness"]
    tempo    = f["tempo"]
    instr    = f["instrumentalness"]
    acoustic = f["acousticness"]
    liveness = f["liveness"]

    # Derived numeric features
    f["duration_min"]               = dur_ms / 60_000.0
    f["optimal_length"]             = 1.0 if 150_000 <= dur_ms <= 240_000 else 0.0
    f["energy_valence"]             = energy * valence
    f["danceability_energy"]        = dance * energy
    f["vocal_instrumental_balance"] = 1.0 - instr
    f["acoustic_energy_balance"]    = acoustic * (1.0 - energy)
    f["is_live"]                    = 1.0 if liveness > 0.8 else 0.0
    f["highly_danceable"]           = 1.0 if dance > 0.7 else 0.0
    f["dance_tempo"]                = dance * tempo
    f["tempo_normalized"]           = max(0.0, min(1.0, (tempo - 60.0) / 120.0))
    f["loudness_normalized"]        = max(0.0, min(1.0, (loudness + 60.0) / 60.0))
    f["is_loud"]                    = 1.0 if loudness > -8.0 else 0.0
    f["party_index"]                = (dance + energy + valence) / 3.0
    f["chill_index"]                = (acoustic + (1.0 - energy) + (1.0 - dance)) / 3.0
    f["workout_index"]              = (energy + f["tempo_normalized"] + dance) / 3.0

    # Date features — use today for a "new track" scenario
    today = datetime.date.today()
    f["release_year"]         = float(today.year)
    f["release_month"]        = float(today.month)
    f["release_day_of_week"]  = float(today.weekday())
    f["is_weekend_release"]   = 1.0 if today.weekday() >= 5 else 0.0
    f["song_age_years"]       = 0.0

    # Artist-level defaults (established artist profile)
    f["artist_avg_popularity"]  = 65.0
    f["artist_song_count"]      = 12.0
    f["artist_max_popularity"]  = 78.0
    f["artist_std_popularity"]  = 12.0
    f["artist_has_viral_hit"]   = 1.0

    # Target-encoded categoricals (neutral means)
    f["key_target_enc"]             = 0.5
    f["mode_target_enc"]            = 0.55
    f["time_signature_target_enc"]  = 0.5

    # Polynomial interactions
    ln = f["loudness_normalized"]
    f["poly_danceability_x_energy"]              = dance * energy
    f["poly_danceability_x_valence"]             = dance * valence
    f["poly_danceability_x_loudness_normalized"] = dance * ln
    f["poly_energy_x_valence"]                   = energy * valence
    f["poly_energy_x_loudness_normalized"]       = energy * ln
    f["poly_valence_x_loudness_normalized"]      = valence * ln

    # Mood buckets
    happy_energetic = energy >= 0.6 and valence >= 0.6
    sad_calm        = energy <  0.4 and valence <  0.4
    sad_energetic   = energy >= 0.6 and valence <  0.4
    neutral         = not (happy_energetic or sad_calm or sad_energetic)
    f["mood_happy_energetic"] = 1.0 if happy_energetic else 0.0
    f["mood_neutral"]         = 1.0 if neutral         else 0.0
    f["mood_sad_calm"]        = 1.0 if sad_calm        else 0.0
    f["mood_sad_energetic"]   = 1.0 if sad_energetic   else 0.0

    # Duration bucket
    short  = dur_ms < 180_000
    long_  = dur_ms > 300_000
    medium = not short and not long_
    f["duration_category_short"]  = 1.0 if short  else 0.0
    f["duration_category_medium"] = 1.0 if medium else 0.0
    f["duration_category_long"]   = 1.0 if long_  else 0.0

    # Tempo bucket
    f["tempo_category_moderate"]  = 1.0 if 80  <= tempo < 110 else 0.0
    f["tempo_category_fast"]      = 1.0 if 110 <= tempo < 140 else 0.0
    f["tempo_category_very_fast"] = 1.0 if tempo >= 140        else 0.0

    # Popularity bucket (use "high" as default for unknown popularity)
    f["popularity_bucket_medium"] = 0.0
    f["popularity_bucket_high"]   = 1.0
    f["popularity_bucket_viral"]  = 0.0

    return pd.DataFrame([[f[feat] for feat in ALL_FEATURES]], columns=ALL_FEATURES)


# ── Dynamic Hybrid Model class ────────────────────────────────────────────────

class DynamicHybridModel:
    """
    Loads four trained classifiers and a scaler once on instantiation,
    then routes each prediction to the most appropriate model based on
    the song's audio features.

    Parameters
    ----------
    models_dir : str | Path, optional
        Override for the models directory (default: <project_root>/models/).
    """

    #: Human-readable names for each model slot
    MODEL_NAMES = {
        "xgboost":          "XGBoost",
        "random_forest":    "Random Forest",
        "lightgbm":         "LightGBM",
        "stacking_ensemble":"Stacking Ensemble",
    }

    def __init__(self, models_dir: str | Path | None = None) -> None:
        mdir = Path(models_dir) if models_dir else _MODELS_DIR
        self._models: Dict[str, Any] = {}
        self._scaler = None
        self._load_all(mdir)

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_all(self, mdir: Path) -> None:
        files = {
            "xgboost":           "xgboost.pkl",
            "random_forest":     "random_forest.pkl",
            "lightgbm":          "lightgbm.pkl",
            "stacking_ensemble": "stacking_ensemble.pkl",
        }
        missing = []
        for key, fname in files.items():
            path = mdir / fname
            if path.exists():
                self._models[key] = joblib.load(path)
            else:
                missing.append(str(path))

        scaler_path = mdir / "scaler_standard.pkl"
        if scaler_path.exists():
            self._scaler = joblib.load(scaler_path)
        else:
            missing.append(str(scaler_path))

        if missing:
            raise FileNotFoundError(
                "Dynamic Hybrid Model: the following required files were not found:\n"
                + "\n".join(f"  • {p}" for p in missing)
            )

    # ── Routing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _select_model_key(features: Dict[str, Any]) -> str:
        """
        Apply the routing rules and return the internal model key.

        Rules (evaluated in order):
          1. danceability > 0.75 AND energy > 0.70  →  XGBoost
          2. acousticness  > 0.60                   →  Random Forest
          3. instrumentalness > 0.50                →  LightGBM
          4. (default)                              →  Stacking Ensemble
        """
        dance  = float(features.get("danceability", 0.0))
        energy = float(features.get("energy", 0.0))
        acoust = float(features.get("acousticness", 0.0))
        instr  = float(features.get("instrumentalness", 0.0))

        if dance > 0.75 and energy > 0.70:
            return "xgboost"
        elif acoust > 0.60:
            return "random_forest"
        elif instr > 0.50:
            return "lightgbm"
        else:
            return "stacking_ensemble"

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_dynamic(self, song_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict virality for a single song using the dynamically selected model.

        Parameters
        ----------
        song_features : dict
            Dictionary of raw audio features. Recognised keys:
            danceability, energy, loudness, speechiness, acousticness,
            instrumentalness, liveness, valence, tempo, duration_ms,
            key, mode, explicit, time_signature.
            Missing keys default to zero / safe neutral values.

        Returns
        -------
        dict
            {
                "prediction"  : int   — 1 = viral, 0 = not viral,
                "probability" : float — viral probability in [0, 1],
                "model_used"  : str   — human-readable model name,
                "model_key"   : str   — internal key (e.g. "xgboost"),
                "routing_reason": str — which rule triggered the selection,
            }
        """
        model_key   = self._select_model_key(song_features)
        model       = self._models[model_key]
        model_name  = self.MODEL_NAMES[model_key]

        # Build routing explanation
        dance  = float(song_features.get("danceability", 0.0))
        energy = float(song_features.get("energy", 0.0))
        acoust = float(song_features.get("acousticness", 0.0))
        instr  = float(song_features.get("instrumentalness", 0.0))

        if model_key == "xgboost":
            reason = (
                f"danceability ({dance:.2f}) > 0.75 "
                f"AND energy ({energy:.2f}) > 0.70"
            )
        elif model_key == "random_forest":
            reason = f"acousticness ({acoust:.2f}) > 0.60"
        elif model_key == "lightgbm":
            reason = f"instrumentalness ({instr:.2f}) > 0.50"
        else:
            reason = "default fallback (no specific rule matched)"

        # Build feature vector and scale
        X    = _build_feature_vector(song_features)
        X_sc = self._scaler.transform(X)

        # Predict
        prediction = int(model.predict(X_sc)[0])
        proba_arr  = model.predict_proba(X_sc)[0]
        probability = float(proba_arr[1]) if len(proba_arr) > 1 else float(proba_arr[0])

        return {
            "prediction":     prediction,
            "probability":    round(probability, 6),
            "model_used":     model_name,
            "model_key":      model_key,
            "routing_reason": reason,
        }

    def __repr__(self) -> str:  # pragma: no cover
        loaded = list(self._models.keys())
        return f"DynamicHybridModel(models_loaded={loaded})"


# ── Module-level convenience function ─────────────────────────────────────────

_default_instance: DynamicHybridModel | None = None


def predict_dynamic(song_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Module-level convenience wrapper.
    Loads a singleton DynamicHybridModel on first call, then reuses it.

    Parameters
    ----------
    song_features : dict  — raw audio feature dictionary (see DynamicHybridModel)

    Returns
    -------
    dict  — {"prediction": int, "probability": float, "model_used": str, ...}

    Example
    -------
    >>> from src.dynamic_hybrid_model import predict_dynamic
    >>> result = predict_dynamic({
    ...     "danceability": 0.85, "energy": 0.80, "loudness": -5.0,
    ...     "speechiness": 0.10, "acousticness": 0.05,
    ...     "instrumentalness": 0.00, "liveness": 0.12,
    ...     "valence": 0.60, "tempo": 120.0, "duration_ms": 210000,
    ... })
    >>> print(result)
    {'prediction': 0, 'probability': 0.009, 'model_used': 'XGBoost', ...}
    """
    global _default_instance
    if _default_instance is None:
        _default_instance = DynamicHybridModel()
    return _default_instance.predict_dynamic(song_features)
