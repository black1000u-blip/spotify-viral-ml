"""
Model and data loading utilities.
All paths are resolved relative to the project root (parent of frontend/).
"""

import os
import joblib
import json
import pandas as pd
import numpy as np
import streamlit as st

# Resolve project root (two levels up from this file: components/ -> frontend/ -> root)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def get_path(*parts):
    """Construct an absolute path relative to the project root."""
    return os.path.join(ROOT, *parts)


@st.cache_resource(show_spinner=False)
def load_model(filename):
    """Load a joblib-serialised model from models/."""
    path = get_path('models', filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_resource(show_spinner=False)
def load_scaler():
    """Load the standard scaler used during training."""
    return load_model('scaler_standard.pkl')


@st.cache_resource(show_spinner=False)
def load_ensemble():
    """Load the stacking ensemble classifier."""
    return load_model('stacking_ensemble.pkl')


@st.cache_data(show_spinner=False)
def load_features_csv():
    """Load the engineered features dataset."""
    path = get_path('data', 'processed', 'features.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_raw_csv():
    """Load the raw Spotify songs dataset."""
    path = get_path('data', 'raw', 'spotify_songs.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_model_results():
    """Load classification model comparison CSV."""
    path = get_path('models', 'model_results.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_regression_results():
    """Load regression model comparison CSV."""
    path = get_path('models', 'regression_results.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_cv_results():
    """Load cross-validation results JSON."""
    path = get_path('models', 'cv_results.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_optimal_thresholds():
    """Load optimal classification thresholds JSON."""
    path = get_path('models', 'optimal_thresholds.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def get_viz_path(filename):
    """Return absolute path to a visualisation image."""
    return get_path('visualizations', filename)


def get_shap_path(filename):
    """Return absolute path to a SHAP image inside models/shap/."""
    return get_path('models', 'shap', filename)


def list_shap_images():
    """Return a sorted list of PNG filenames inside models/shap/."""
    shap_dir = get_path('models', 'shap')
    if os.path.exists(shap_dir):
        return sorted(f for f in os.listdir(shap_dir) if f.endswith('.png'))
    return []


def detect_column(df: pd.DataFrame, keyword: str):
    """Find the first column whose name contains `keyword` (case-insensitive)."""
    for c in df.columns:
        if keyword.lower() in c.lower():
            return c
    return None
