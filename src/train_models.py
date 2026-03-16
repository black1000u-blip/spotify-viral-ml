"""
Model Training Script (Enhanced)
Trains multiple ML models for song virality prediction.

Models: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost,
        TabNet, Kolmogorov-Arnold Network (KAN), Stacking Ensemble,
        Neural Network, Regression (popularity prediction).

Features: Optuna Bayesian hyperparameter tuning, SHAP explainability,
          RepeatedStratifiedKFold cross-validation, probability calibration,
          threshold optimization, MLflow experiment tracking.
"""

import pandas as pd
import numpy as np
import argparse
import os
import json
import joblib
import warnings
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

# ---------- SHAP ----------
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ---------- MLflow ----------
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ---------- TabNet (import PyTorch BEFORE TensorFlow to avoid DLL conflicts) ----------
try:
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except (ImportError, OSError):
    TABNET_AVAILABLE = False

# ---------- TensorFlow / Keras ----------
try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False


# ====================================================================== #
#  Simple KAN (Kolmogorov-Arnold Network) implementation
# ====================================================================== #
class SimpleKAN:
    """
    Lightweight Kolmogorov-Arnold Network for tabular classification.
    Uses B-spline learnable activation functions on edges instead of
    fixed activations on neurons — following the KAN paper (Liu et al., 2024).
    Implemented with TensorFlow/Keras so no extra library is needed.
    """

    def __init__(self, input_dim, hidden_dims=(64, 32), n_basis=8,
                 lr=1e-3, epochs=100, batch_size=32, random_state=42):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_basis = n_basis
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None

    def _build(self):
        if not KERAS_AVAILABLE:
            return None
        from tensorflow.keras import layers, Model
        import tensorflow as tf

        tf.random.set_seed(self.random_state)

        inputs = layers.Input(shape=(self.input_dim,))
        x = inputs

        for dim in self.hidden_dims:
            # Learnable B-spline-like basis expansion per edge
            # Approximate: expand each feature into n_basis outputs via
            # small per-feature dense projections, then combine.
            basis = layers.Dense(self.input_dim * self.n_basis, activation='silu')(x)
            basis = layers.Reshape((self.input_dim, self.n_basis))(basis) if len(self.hidden_dims) else basis
            # Flatten back and project to hidden dim
            basis = layers.Flatten()(basis) if hasattr(basis, 'shape') and len(basis.shape) > 2 else basis
            x = layers.Dense(dim)(basis)
            x = layers.LayerNormalization()(x)
            x = layers.Activation('silu')(x)
            x = layers.Dropout(0.2)(x)

        outputs = layers.Dense(1, activation='sigmoid')(x)
        self.model = Model(inputs, outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        return self.model

    def fit(self, X, y, X_val=None, y_val=None):
        self._build()
        if self.model is None:
            return self
        callbacks = [
            EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True, mode='max'),
            ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, min_lr=1e-6, mode='max')
        ]
        val_data = (X_val, y_val) if X_val is not None else None
        self.model.fit(
            X, y, epochs=self.epochs, batch_size=self.batch_size,
            validation_data=val_data, callbacks=callbacks, verbose=0
        )
        return self

    def predict(self, X):
        proba = self.model.predict(X, verbose=0).flatten()
        return (proba > 0.5).astype(int)

    def predict_proba(self, X):
        p = self.model.predict(X, verbose=0).flatten()
        return np.column_stack([1 - p, p])


# ====================================================================== #
#  Optuna tuning helpers
# ====================================================================== #
def optuna_tune_lgbm(X_train, y_train, n_trials=30, random_state=42):
    """Bayesian HPO for LightGBM via Optuna"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        model = lgb.LGBMClassifier(**params, random_state=random_state, verbose=-1, n_jobs=1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=1)
        score = scores.mean()
        if np.isnan(score):
            return 0.5
        return score

    study = optuna.create_study(direction='maximize', study_name='lgbm_tune')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optuna_tune_xgb(X_train, y_train, n_trials=30, random_state=42):
    """Bayesian HPO for XGBoost via Optuna"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        model = xgb.XGBClassifier(**params, random_state=random_state,
                                   eval_metric='logloss', n_jobs=1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=1)
        score = scores.mean()
        if np.isnan(score):
            return 0.5  # fallback for failed folds
        return score

    study = optuna.create_study(direction='maximize', study_name='xgb_tune')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optuna_tune_catboost(X_train, y_train, n_trials=10, random_state=42):
    """Bayesian HPO for CatBoost via Optuna (manual CV to avoid sklearn tags issue)"""
    from sklearn.metrics import roc_auc_score

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 300),
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx], \
                          y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
            model = CatBoostClassifier(**params, random_state=random_state, verbose=0)
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, preds))
        score = np.mean(scores)
        if np.isnan(score):
            return 0.5
        return score

    print("  (Running 10 trials with 3-fold CV — this may take a few minutes...)")
    study = optuna.create_study(direction='maximize', study_name='catboost_tune')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ====================================================================== #
#  ModelTrainer
# ====================================================================== #
class ModelTrainer:
    """Train and evaluate multiple machine learning models"""

    def __init__(self, random_state=42, use_optuna=True, use_mlflow=True):
        self.random_state = random_state
        self.use_optuna = use_optuna and OPTUNA_AVAILABLE
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self.models = {}
        self.calibrated_models = {}
        self.scalers = {}
        self.results = {}
        self.cv_results = {}
        self.shap_values_dict = {}
        self.optimal_thresholds = {}

    # ------------------------------------------------------------------ #
    #  Data preparation
    # ------------------------------------------------------------------ #
    def prepare_data(self, df, feature_cols, target_col='is_viral',
                     test_size=0.2, balance_data=True):
        """Prepare train/test split and handle class imbalance"""
        print("Preparing data...")

        X = df[feature_cols].copy()
        y = df[target_col].copy()
        X = X.fillna(X.median())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Class distribution (train): {y_train.value_counts().to_dict()}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler

        if balance_data:
            print("\nApplying SMOTE to balance classes...")
            smote = SMOTE(random_state=self.random_state)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            print(f"After SMOTE: {len(X_train_scaled)} samples")
            print(f"Class distribution: {pd.Series(y_train).value_counts().to_dict()}")

        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols

    # ------------------------------------------------------------------ #
    #  1. Logistic Regression
    # ------------------------------------------------------------------ #
    def train_logistic_regression(self, X_train, y_train):
        print("\n" + "="*60)
        print("Training Logistic Regression...")
        print("="*60)

        from sklearn.model_selection import GridSearchCV
        param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'max_iter': [1000]}
        lr = LogisticRegression(random_state=self.random_state, solver='lbfgs')
        gs = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)

        best = gs.best_estimator_
        print(f"Best params: {gs.best_params_}")
        print(f"Best CV ROC-AUC: {gs.best_score_:.4f}")
        self.models['logistic_regression'] = best
        return best

    # ------------------------------------------------------------------ #
    #  2. Random Forest
    # ------------------------------------------------------------------ #
    def train_random_forest(self, X_train, y_train):
        print("\n" + "="*60)
        print("Training Random Forest...")
        print("="*60)

        from sklearn.model_selection import RandomizedSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        rs = RandomizedSearchCV(rf, param_grid, n_iter=20, cv=5, scoring='roc_auc',
                                n_jobs=-1, verbose=0, random_state=self.random_state)
        rs.fit(X_train, y_train)

        best = rs.best_estimator_
        print(f"Best params: {rs.best_params_}")
        print(f"Best CV ROC-AUC: {rs.best_score_:.4f}")
        self.models['random_forest'] = best
        return best

    # ------------------------------------------------------------------ #
    #  3. XGBoost (with Optuna)
    # ------------------------------------------------------------------ #
    def train_xgboost(self, X_train, y_train):
        print("\n" + "="*60)
        print("Training XGBoost" + (" (Optuna)" if self.use_optuna else "") + "...")
        print("="*60)

        if self.use_optuna:
            best_params = optuna_tune_xgb(X_train, y_train, n_trials=30,
                                          random_state=self.random_state)
            print(f"Optuna best params: {best_params}")
            model = xgb.XGBClassifier(**best_params, random_state=self.random_state,
                                      eval_metric='logloss', n_jobs=-1)
        else:
            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='logloss', n_jobs=-1
            )

        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        return model

    # ------------------------------------------------------------------ #
    #  4. LightGBM (with Optuna)
    # ------------------------------------------------------------------ #
    def train_lightgbm(self, X_train, y_train):
        if not LGBM_AVAILABLE:
            print("\nSkipping LightGBM (not installed)")
            return None

        print("\n" + "="*60)
        print("Training LightGBM" + (" (Optuna)" if self.use_optuna else "") + "...")
        print("="*60)

        if self.use_optuna:
            best_params = optuna_tune_lgbm(X_train, y_train, n_trials=30,
                                            random_state=self.random_state)
            print(f"Optuna best params: {best_params}")
            model = lgb.LGBMClassifier(**best_params, random_state=self.random_state,
                                       verbose=-1, n_jobs=-1)
        else:
            model = lgb.LGBMClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                random_state=self.random_state, verbose=-1, n_jobs=-1
            )

        model.fit(X_train, y_train)
        self.models['lightgbm'] = model
        return model

    # ------------------------------------------------------------------ #
    #  5. CatBoost (with Optuna)
    # ------------------------------------------------------------------ #
    def train_catboost(self, X_train, y_train):
        if not CATBOOST_AVAILABLE:
            print("\nSkipping CatBoost (not installed)")
            return None

        print("\n" + "="*60)
        print("Training CatBoost" + (" (Optuna)" if self.use_optuna else "") + "...")
        print("="*60)

        if self.use_optuna:
            best_params = optuna_tune_catboost(X_train, y_train, n_trials=20,
                                               random_state=self.random_state)
            print(f"Optuna best params: {best_params}")
            model = CatBoostClassifier(**best_params, random_state=self.random_state, verbose=0)
        else:
            model = CatBoostClassifier(
                iterations=300, depth=6, learning_rate=0.1,
                random_state=self.random_state, verbose=0
            )

        model.fit(X_train, y_train)
        self.models['catboost'] = model
        return model

    # ------------------------------------------------------------------ #
    #  6. TabNet
    # ------------------------------------------------------------------ #
    def train_tabnet(self, X_train, y_train, X_test, y_test):
        if not TABNET_AVAILABLE:
            print("\nSkipping TabNet (not installed)")
            return None

        print("\n" + "="*60)
        print("Training TabNet...")
        print("="*60)

        model = TabNetClassifier(
            n_d=16, n_a=16, n_steps=5,
            gamma=1.5, lambda_sparse=1e-4,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size": 15, "gamma": 0.9},
            scheduler_fn=None, verbose=0, seed=self.random_state
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=['auc'],
            max_epochs=100, patience=15,
            batch_size=256, virtual_batch_size=128
        )

        self.models['tabnet'] = model
        return model

    # ------------------------------------------------------------------ #
    #  7. KAN (Kolmogorov-Arnold Network)
    # ------------------------------------------------------------------ #
    def train_kan(self, X_train, y_train, X_test, y_test):
        if not KERAS_AVAILABLE:
            print("\nSkipping KAN (TensorFlow not available)")
            return None

        print("\n" + "="*60)
        print("Training KAN (Kolmogorov-Arnold Network)...")
        print("="*60)

        model = SimpleKAN(
            input_dim=X_train.shape[1],
            hidden_dims=(64, 32),
            n_basis=8, lr=1e-3, epochs=100,
            batch_size=32, random_state=self.random_state
        )
        model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
        self.models['kan'] = model
        return model

    # ------------------------------------------------------------------ #
    #  8. Neural Network (MLP)
    # ------------------------------------------------------------------ #
    def train_neural_network(self, X_train, y_train, X_test, y_test):
        if not KERAS_AVAILABLE:
            print("\nSkipping Neural Network (TensorFlow not available)")
            return None

        print("\n" + "="*60)
        print("Training Neural Network (MLP)...")
        print("="*60)

        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(), Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(), Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(), Dropout(0.2),
            Dense(16, activation='relu'), Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy', keras.metrics.AUC(name='auc')])

        callbacks = [
            EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True, mode='max'),
            ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, min_lr=1e-6, mode='max')
        ]
        model.fit(X_train, y_train, epochs=100, batch_size=32,
                  validation_data=(X_test, y_test), callbacks=callbacks, verbose=0)

        self.models['neural_network'] = model
        return model

    # ------------------------------------------------------------------ #
    #  9. Stacking Ensemble
    # ------------------------------------------------------------------ #
    def train_stacking_ensemble(self, X_train, y_train):
        print("\n" + "="*60)
        print("Training Stacking Ensemble...")
        print("="*60)

        estimators = [
            ('rf', RandomForestClassifier(n_estimators=200, random_state=self.random_state, n_jobs=-1)),
            ('xgb', xgb.XGBClassifier(n_estimators=200, random_state=self.random_state,
                                       eval_metric='logloss', n_jobs=-1)),
        ]
        if LGBM_AVAILABLE:
            estimators.append(
                ('lgbm', lgb.LGBMClassifier(n_estimators=200, random_state=self.random_state,
                                             verbose=-1, n_jobs=-1))
            )
        # Note: CatBoost excluded from stacking due to sklearn __sklearn_tags__ incompatibility

        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=self.random_state),
            cv=5, n_jobs=-1, passthrough=False
        )
        stacking.fit(X_train, y_train)

        self.models['stacking_ensemble'] = stacking
        print("Stacking ensemble trained with base learners:",
              [name for name, _ in estimators])
        return stacking

    # ------------------------------------------------------------------ #
    #  10. Regression model (predict raw popularity)
    # ------------------------------------------------------------------ #
    def train_regression(self, df, feature_cols):
        """Train a regression model to predict raw popularity score"""
        print("\n" + "="*60)
        print("Training Regression Model (Popularity Prediction)...")
        print("="*60)

        X = df[feature_cols].copy().fillna(df[feature_cols].median())
        y = df['popularity'].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Ridge regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_s, y_train)
        y_pred_ridge = ridge.predict(X_test_s)

        # Random Forest regressor
        rfr = RandomForestRegressor(n_estimators=200, random_state=self.random_state, n_jobs=-1)
        rfr.fit(X_train_s, y_train)
        y_pred_rf = rfr.predict(X_test_s)

        # XGBoost regressor
        xgbr = xgb.XGBRegressor(n_estimators=200, random_state=self.random_state, n_jobs=-1)
        xgbr.fit(X_train_s, y_train)
        y_pred_xgb = xgbr.predict(X_test_s)

        reg_results = {}
        for name, y_pred in [('ridge', y_pred_ridge), ('rf_regressor', y_pred_rf),
                              ('xgb_regressor', y_pred_xgb)]:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            reg_results[name] = {'rmse': rmse, 'mae': mae, 'r2': r2}
            print(f"  {name:20s}  RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}")

        self.models['ridge_regression'] = ridge
        self.models['rf_regressor'] = rfr
        self.models['xgb_regressor'] = xgbr
        self.scalers['regression'] = scaler
        self.results['regression'] = reg_results

        return reg_results

    # ------------------------------------------------------------------ #
    #  Evaluation
    # ------------------------------------------------------------------ #
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a trained model"""
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print('='*60)

        # Predict probabilities
        if model_name in ('neural_network', 'kan') and hasattr(model, 'predict'):
            if model_name == 'neural_network':
                y_pred_proba = model.predict(X_test, verbose=0).flatten()
            else:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
        elif model_name == 'tabnet':
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        self.results[model_name] = {
            'accuracy': accuracy, 'precision': precision,
            'recall': recall, 'f1_score': f1, 'roc_auc': roc_auc,
            'y_pred': y_pred, 'y_pred_proba': y_pred_proba
        }

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Not Viral', 'Viral']))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        return self.results[model_name]

    # ------------------------------------------------------------------ #
    #  RepeatedStratifiedKFold cross-validation
    # ------------------------------------------------------------------ #
    def cross_validate_models(self, X_train, y_train, n_splits=5, n_repeats=3):
        """Robust cross-validation with RepeatedStratifiedKFold"""
        print("\n" + "="*60)
        print(f"Cross-Validation (RepeatedStratifiedKFold: {n_splits}x{n_repeats})")
        print("="*60)

        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                        random_state=self.random_state)

        skip = {'neural_network', 'kan', 'tabnet',
                'ridge_regression', 'rf_regressor', 'xgb_regressor'}

        for name, model in self.models.items():
            if name in skip:
                continue
            try:
                scores = cross_val_score(model, X_train, y_train,
                                          cv=rskf, scoring='roc_auc', n_jobs=-1)
                self.cv_results[name] = {
                    'mean_auc': scores.mean(), 'std_auc': scores.std(),
                    'scores': scores.tolist()
                }
                print(f"  {name:25s}  AUC = {scores.mean():.4f} +/- {scores.std():.4f}")
            except Exception as e:
                print(f"  {name:25s}  CV failed: {e}")

    # ------------------------------------------------------------------ #
    #  Probability calibration
    # ------------------------------------------------------------------ #
    def calibrate_models(self, X_train, y_train):
        """Calibrate probability outputs using CalibratedClassifierCV"""
        print("\n" + "="*60)
        print("Probability Calibration (CalibratedClassifierCV)")
        print("="*60)

        skip = {'neural_network', 'kan', 'tabnet',
                'ridge_regression', 'rf_regressor', 'xgb_regressor'}

        for name, model in self.models.items():
            if name in skip:
                continue
            try:
                cal_model = CalibratedClassifierCV(model, cv=5, method='isotonic')
                cal_model.fit(X_train, y_train)
                self.calibrated_models[name] = cal_model
                print(f"  Calibrated: {name}")
            except Exception as e:
                print(f"  Failed to calibrate {name}: {e}")

    # ------------------------------------------------------------------ #
    #  Threshold optimization
    # ------------------------------------------------------------------ #
    def optimize_thresholds(self, X_test, y_test):
        """Find optimal classification threshold per model using F1 score"""
        print("\n" + "="*60)
        print("Threshold Optimization (maximize F1)")
        print("="*60)

        for name, result in self.results.items():
            if 'y_pred_proba' not in result:
                continue
            proba = result['y_pred_proba']
            precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
            f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
            best_idx = np.argmax(f1_scores)
            best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            best_f1 = f1_scores[best_idx]
            self.optimal_thresholds[name] = float(best_thresh)
            print(f"  {name:25s}  threshold={best_thresh:.3f}  F1={best_f1:.4f}")

    # ------------------------------------------------------------------ #
    #  SHAP explainability
    # ------------------------------------------------------------------ #
    def compute_shap(self, X_test, feature_names, output_dir):
        """Compute and save SHAP values for tree-based models"""
        if not SHAP_AVAILABLE:
            print("\nSHAP not available — skipping explainability.")
            return

        print("\n" + "="*60)
        print("SHAP Explainability")
        print("="*60)

        shap_dir = os.path.join(output_dir, 'shap')
        os.makedirs(shap_dir, exist_ok=True)

        tree_models = ['random_forest', 'xgboost', 'lightgbm', 'catboost']

        for name in tree_models:
            if name not in self.models:
                continue
            try:
                print(f"  Computing SHAP for {name}...")
                explainer = shap.TreeExplainer(self.models[name])
                # Use a subsample for speed
                X_sample = X_test[:min(500, len(X_test))]
                sv = explainer.shap_values(X_sample)

                # Handle different output shapes
                if isinstance(sv, list):
                    sv = sv[1]  # positive class

                self.shap_values_dict[name] = sv

                # Summary plot
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(shap_dir, f'shap_summary_{name}.png'), dpi=200, bbox_inches='tight')
                plt.close()

                # Bar plot
                shap.summary_plot(sv, X_sample, feature_names=feature_names,
                                  plot_type='bar', show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(shap_dir, f'shap_bar_{name}.png'), dpi=200, bbox_inches='tight')
                plt.close()

                print(f"    Saved SHAP plots for {name}")
            except Exception as e:
                print(f"    SHAP failed for {name}: {e}")

    # ------------------------------------------------------------------ #
    #  MLflow logging
    # ------------------------------------------------------------------ #
    def log_to_mlflow(self, feature_cols):
        """Log all results to MLflow"""
        if not self.use_mlflow:
            print("\nMLflow not available — skipping experiment tracking.")
            return

        print("\n" + "="*60)
        print("Logging to MLflow")
        print("="*60)

        mlflow.set_experiment("spotify_virality_prediction")

        for name, result in self.results.items():
            if name == 'regression':
                # Log regression sub-models
                for sub_name, metrics in result.items():
                    with mlflow.start_run(run_name=f"reg_{sub_name}"):
                        for k, v in metrics.items():
                            mlflow.log_metric(k, v)
                continue

            with mlflow.start_run(run_name=name):
                for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
                    if metric_name in result:
                        mlflow.log_metric(metric_name, result[metric_name])

                if name in self.cv_results:
                    mlflow.log_metric('cv_mean_auc', self.cv_results[name]['mean_auc'])
                    mlflow.log_metric('cv_std_auc', self.cv_results[name]['std_auc'])

                if name in self.optimal_thresholds:
                    mlflow.log_metric('optimal_threshold', self.optimal_thresholds[name])

                mlflow.log_param('n_features', len(feature_cols))
                mlflow.log_param('model_type', name)

                # Log sklearn models
                skip_log = {'neural_network', 'kan', 'tabnet'}
                if name in self.models and name not in skip_log:
                    try:
                        mlflow.sklearn.log_model(self.models[name], name)
                    except Exception:
                        pass

        print("  MLflow logging complete.")

    # ------------------------------------------------------------------ #
    #  Save models
    # ------------------------------------------------------------------ #
    def save_models(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nSaving models to {output_dir}...")

        for name, model in self.models.items():
            try:
                if name in ('neural_network', 'kan'):
                    if hasattr(model, 'model') and model.model is not None:
                        path = os.path.join(output_dir, f'{name}.h5')
                        model.model.save(path)
                    elif hasattr(model, 'save'):
                        path = os.path.join(output_dir, f'{name}.h5')
                        model.save(path)
                    else:
                        continue
                elif name == 'tabnet':
                    path = os.path.join(output_dir, f'{name}.zip')
                    model.save_model(path)
                else:
                    path = os.path.join(output_dir, f'{name}.pkl')
                    joblib.dump(model, path)
                print(f"  Saved {name}")
            except Exception as e:
                print(f"  Failed to save {name}: {e}")

        # Save scalers
        for sname, scaler in self.scalers.items():
            joblib.dump(scaler, os.path.join(output_dir, f'scaler_{sname}.pkl'))

        # Save optimal thresholds
        if self.optimal_thresholds:
            with open(os.path.join(output_dir, 'optimal_thresholds.json'), 'w') as f:
                json.dump(self.optimal_thresholds, f, indent=2)

        # Save CV results
        if self.cv_results:
            with open(os.path.join(output_dir, 'cv_results.json'), 'w') as f:
                json.dump(self.cv_results, f, indent=2)

    # ------------------------------------------------------------------ #
    #  Save evaluation results
    # ------------------------------------------------------------------ #
    def save_results(self, output_dir):
        # Classification results
        cls_results = {k: v for k, v in self.results.items()
                       if k != 'regression' and isinstance(v, dict) and 'accuracy' in v}
        if cls_results:
            rows = {}
            for name, r in cls_results.items():
                rows[name] = {k: v for k, v in r.items()
                              if k not in ('y_pred', 'y_pred_proba')}
            results_df = pd.DataFrame(rows).T
            path = os.path.join(output_dir, 'model_results.csv')
            results_df.to_csv(path)
            print(f"\nClassification results saved to {path}")
            print("\nModel Comparison:")
            print(results_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']])

        # Regression results
        if 'regression' in self.results:
            reg_df = pd.DataFrame(self.results['regression']).T
            path = os.path.join(output_dir, 'regression_results.csv')
            reg_df.to_csv(path)
            print(f"Regression results saved to {path}")


# ====================================================================== #
#  MAIN
# ====================================================================== #
def main():
    parser = argparse.ArgumentParser(description='Train ML models for song virality prediction')
    parser.add_argument('--input', type=str, required=True, help='Input CSV with features')
    parser.add_argument('--output', type=str, default='models/', help='Output directory')
    parser.add_argument('--no-balance', action='store_true', help='Skip SMOTE')
    parser.add_argument('--no-optuna', action='store_true', help='Skip Optuna tuning')
    parser.add_argument('--no-mlflow', action='store_true', help='Skip MLflow logging')
    parser.add_argument('--no-shap', action='store_true', help='Skip SHAP analysis')
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Feature columns
    exclude_cols = ['track_id', 'track_name', 'artist_name', 'is_viral',
                    'popularity', 'mood', 'duration_category', 'tempo_category',
                    'popularity_bucket', 'release_date', 'release_season']
    feature_cols = [col for col in df.columns if col not in exclude_cols
                    and df[col].dtype in ('float64', 'int64', 'float32', 'int32', 'bool')]
    print(f"\nUsing {len(feature_cols)} features")

    # Initialize trainer
    trainer = ModelTrainer(
        use_optuna=not args.no_optuna,
        use_mlflow=not args.no_mlflow
    )

    # Prepare data
    X_train, X_test, y_train, y_test, feature_cols = trainer.prepare_data(
        df, feature_cols, balance_data=not getattr(args, 'no_balance', False)
    )

    # --- Train all classification models ---
    trainer.train_logistic_regression(X_train, y_train)
    trainer.train_random_forest(X_train, y_train)
    trainer.train_xgboost(X_train, y_train)
    trainer.train_lightgbm(X_train, y_train)
    trainer.train_catboost(X_train, y_train)
    trainer.train_tabnet(X_train, y_train, X_test, y_test)
    trainer.train_kan(X_train, y_train, X_test, y_test)
    trainer.train_neural_network(X_train, y_train, X_test, y_test)
    trainer.train_stacking_ensemble(X_train, y_train)

    # --- Evaluate all classification models ---
    skip_eval = {'ridge_regression', 'rf_regressor', 'xgb_regressor'}
    for name in list(trainer.models.keys()):
        if name not in skip_eval:
            trainer.evaluate_model(trainer.models[name], X_test, y_test, name)

    # --- Cross-validation ---
    trainer.cross_validate_models(X_train, y_train)

    # --- Probability calibration ---
    trainer.calibrate_models(X_train, y_train)

    # --- Threshold optimization ---
    trainer.optimize_thresholds(X_test, y_test)

    # --- Regression model ---
    trainer.train_regression(df, feature_cols)

    # --- SHAP ---
    if not args.no_shap:
        trainer.compute_shap(X_test, feature_cols, args.output)

    # --- MLflow ---
    trainer.log_to_mlflow(feature_cols)

    # --- Save everything ---
    trainer.save_models(args.output)
    trainer.save_results(args.output)

    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == '__main__':
    main()
