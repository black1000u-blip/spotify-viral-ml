"""
Model and data loading utilities.
Now with MongoDB integration for persistent result retrieval.
"""

import os
import joblib
import json
import pandas as pd
import numpy as np
import streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Resolve project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# MongoDB Config
MONGO_URI = os.getenv('MONGO_URI', "mongodb+srv://Dradmin:Mongo%40db%23123@cluster0.qa3itof.mongodb.net/")
DB_NAME = "spotify_data"

@st.cache_resource(show_spinner=False)
def get_mongo_db():
    """Get active MongoDB database connection"""
    try:
        client = MongoClient(MONGO_URI)
        return client[DB_NAME]
    except Exception:
        return None

def get_path(*parts):
    """Construct an absolute path relative to the project root."""
    return os.path.join(ROOT, *parts)

@st.cache_resource(show_spinner=False)
def load_model(filename):
    path = get_path('models', filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_resource(show_spinner=False)
def load_scaler():
    return load_model('scaler_standard.pkl')

@st.cache_resource(show_spinner=False)
def load_ensemble():
    return load_model('stacking_ensemble.pkl')

@st.cache_data(show_spinner=False)
def load_raw_csv():
    """Load the raw Spotify songs dataset."""
    path = get_path('data', 'raw', 'spotify_songs.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_features_csv():
    """Load the engineered features dataset."""
    path = get_path('data', 'processed', 'features.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_model_results():
    """Load classification results from MongoDB (preferred) or local CSV"""
    db = get_mongo_db()
    if db is not None:
        try:
            doc = db['model_performance'].find_one(sort=[('timestamp', -1)])
            if doc and 'classification_results' in doc:
                df = pd.DataFrame(doc['classification_results']).T
                df = df.reset_index().rename(columns={'index': 'Model'})
                return df
        except Exception:
            pass
            
    # Fallback to local file
    path = get_path('models', 'model_results.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_regression_results():
    path = get_path('models', 'regression_results.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_cv_results():
    path = get_path('models', 'cv_results.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data(show_spinner=False)
def load_optimal_thresholds():
    db = get_mongo_db()
    if db is not None:
        try:
            doc = db['model_performance'].find_one(sort=[('timestamp', -1)])
            if doc and 'optimal_thresholds' in doc:
                return doc['optimal_thresholds']
        except Exception:
            pass
            
    path = get_path('models', 'optimal_thresholds.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def get_viz_path(filename):
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
    for c in df.columns:
        if keyword.lower() in c.lower():
            return c
    return None
