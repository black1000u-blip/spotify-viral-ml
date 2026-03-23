"""
Model Training Script (Enhanced)
Trains multiple ML models for song virality prediction.
Now integrated with MongoDB for persistent results storage.
"""

import pandas as pd
import numpy as np
import argparse
import os
import json
import joblib
import warnings
from pymongo import MongoClient
import datetime

warnings.filterwarnings('ignore')

# ---------- sklearn ----------
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    RepeatedStratifiedKFold, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier, RandomForestRegressor
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# ---------- LightGBM ----------
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

# ---------- CatBoost ----------
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# ---------- Optuna ----------
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# ====================================================================== #
#  ModelTrainer
# ====================================================================== #
class ModelTrainer:
    """Train and evaluate multiple machine learning models with MongoDB logging"""

    def __init__(self, random_state=42, use_optuna=True, mongo_uri=None):
        self.random_state = random_state
        self.use_optuna = use_optuna and OPTUNA_AVAILABLE
        self.models = {}
        self.calibrated_models = {}
        self.scalers = {}
        self.results = {}
        self.cv_results = {}
        self.optimal_thresholds = {}
        
        # MongoDB Setup
        self.mongo_uri = mongo_uri
        self.db = None
        if self.mongo_uri:
            try:
                self.client = MongoClient(self.mongo_uri)
                self.db = self.client['spotify_data']
                print(f"✅ Connected to MongoDB for training logs: {self.db.name}")
            except Exception as e:
                print(f"❌ MongoDB Connection Error: {e}")

    def prepare_data(self, df, feature_cols, target_col='is_viral', test_size=0.2):
        """Prepare train/test split and handle class imbalance"""
        X = df[feature_cols].copy().fillna(df[feature_cols].median())
        y = df[target_col].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler

        # SMOTE for balance
        minority_size = y_train.value_counts().min()
        if minority_size > 5:
            smote = SMOTE(random_state=self.random_state)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            print(f"Applied SMOTE: {len(X_train_scaled)} samples")

        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols

    def train_all(self, X_train, y_train, X_test, y_test):
        """Train standard suite of models"""
        # 1. Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=self.random_state)
        lr.fit(X_train, y_train)
        self.models['logistic_regression'] = lr

        # 2. Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf

        # 3. XGBoost
        xg = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=self.random_state, n_jobs=-1)
        xg.fit(X_train, y_train)
        self.models['xgboost'] = xg

        # 4. LightGBM
        if LGBM_AVAILABLE:
            lg = lgb.LGBMClassifier(n_estimators=100, random_state=self.random_state, verbose=-1, n_jobs=-1)
            lg.fit(X_train, y_train)
            self.models['lightgbm'] = lg

        # 5. CatBoost
        if CATBOOST_AVAILABLE:
            cb = CatBoostClassifier(iterations=100, random_state=self.random_state, verbose=0)
            cb.fit(X_train, y_train)
            self.models['catboost'] = cb

        # 6. Stacking Ensemble
        base_models = [('rf', rf), ('xgb', xg)]
        if LGBM_AVAILABLE: base_models.append(('lgbm', lg))
        stack = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=5)
        stack.fit(X_train, y_train)
        self.models['stacking_ensemble'] = stack

    def evaluate_all(self, X_test, y_test):
        """Evaluate all trained models"""
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            self.results[name] = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
                'roc_auc': float(roc_auc_score(y_test, y_prob))
            }
            print(f"  {name:20s}: AUC={self.results[name]['roc_auc']:.4f}")

    def save_results_to_mongo(self):
        """Save performance metrics to MongoDB"""
        if self.db is not None:
            try:
                collection = self.db['model_performance']
                # Clear old results
                collection.delete_many({})
                
                doc = {
                    'timestamp': datetime.datetime.now(),
                    'classification_results': self.results,
                    'optimal_thresholds': self.optimal_thresholds,
                    'feature_count': len(self.scalers['standard'].feature_names_in_) if 'standard' in self.scalers else 0
                }
                collection.insert_one(doc)
                print(f"✅ Training results saved to MongoDB (model_performance)")
            except Exception as e:
                print(f"❌ MongoDB Save Error: {e}")

    def save_local(self, output_dir):
        """Save models and scalers locally"""
        os.makedirs(output_dir, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(output_dir, f"{name}.pkl"))
        if 'standard' in self.scalers:
            joblib.dump(self.scalers['standard'], os.path.join(output_dir, "scaler_standard.pkl"))
        
        # Save results CSV for frontend fallback
        res_df = pd.DataFrame(self.results).T
        res_df.to_csv(os.path.join(output_dir, "model_results.csv"))

def main():
    parser = argparse.ArgumentParser(description='Train Spotify virality models')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='models/')
    parser.add_argument('--mongo_uri', type=str, default="mongodb+srv://Dradmin:Mongo%40db%23123@cluster0.qa3itof.mongodb.net/")
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)
    # Define features to explicitly ignore for training (leakage or redundant)
    ignore_cols = [
        'is_viral', 'popularity', 'duration_ms', 'key', 'mode', 'time_signature',
        'artist_avg_popularity', 'artist_max_popularity'
    ]
    
    # Try to load the feature list saved by feature_engineering.py
    feature_list_path = args.input.replace('.csv', '_features.txt')
    if os.path.exists(feature_list_path):
        with open(feature_list_path, 'r') as f:
            feature_cols = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(feature_cols)} features from {feature_list_path}")
    else:
        # Fallback: Filter to only numeric features and exclude ignore_cols
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if col not in ignore_cols]
        print(f"Using fallback filtering: {len(feature_cols)} features selected")
    
    trainer = ModelTrainer(mongo_uri=args.mongo_uri)
    X_train, X_test, y_train, y_test, feats = trainer.prepare_data(df, feature_cols)
    
    trainer.train_all(X_train, y_train, X_test, y_test)
    trainer.evaluate_all(X_test, y_test)
    trainer.save_results_to_mongo()
    trainer.save_local(args.output)
    
    print("\nTRAINING COMPLETED SUCCESSFULLY!")

if __name__ == '__main__':
    main()
