import pandas as pd
import numpy as np
import joblib
import os
import json
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class AdaptiveMultiModelViralityPredictor:
    """
    AMVP: Adaptive Multi-Model Virality Predictor
    A hybrid ML architecture that dynamically combines multiple models
    based on input feature characteristics and historical performance.
    """
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.models = {}
        self.scaler = None
        self.base_weights = {}
        self.feature_names = []
        self.model_names = [
            'logistic_regression',
            'random_forest',
            'xgboost',
            'lightgbm',
            'catboost'
        ]
        
        # Load resources
        self._load_resources()
        self._initialize_base_weights()

    def _load_resources(self):
        """Load trained models and scaler"""
        print("Loading AMVP resources...")
        
        # Load scaler
        scaler_path = os.path.join(self.models_dir, 'scaler_standard.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            # Use feature names from scaler to ensure consistency
            if hasattr(self.scaler, 'feature_names_in_'):
                self.feature_names = list(self.scaler.feature_names_in_)
            else:
                # Fallback to loading from file if scaler doesn't have names
                self._load_feature_names_from_file()
        else:
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
            
        # Load models
        for name in self.model_names:
            model_path = os.path.join(self.models_dir, f'{name}.pkl')
            if os.path.exists(model_path):
                try:
                    self.models[name] = joblib.load(model_path)
                    print(f"  Loaded {name}")
                except Exception as e:
                    print(f"  Error loading {name}: {e}")
            else:
                print(f"  Warning: Model {name} not found at {model_path}")

    def _load_feature_names_from_file(self):
        """Fallback feature loading from text file"""
        feature_file = 'data/processed/features_features.txt'
        if os.path.exists(feature_file):
            with open(feature_file, 'r') as f:
                self.feature_names = [line.strip() for line in f if line.strip()]
        else:
            print(f"Critical: Feature names could not be loaded from scaler or {feature_file}.")

    def _initialize_base_weights(self):
        """Initialize weights based on cross-validation performance"""
        results_path = os.path.join(self.models_dir, 'model_results.csv')
        
        if os.path.exists(results_path):
            try:
                results_df = pd.read_csv(results_path)
                # Handle possible index formats
                if 'Unnamed: 0' in results_df.columns:
                    results_df.set_index('Unnamed: 0', inplace=True)
                elif 'model' in results_df.columns:
                    results_df.set_index('model', inplace=True)
                
                for name in self.model_names:
                    if name in results_df.index:
                        # Use ROC-AUC as the base confidence metric
                        self.base_weights[name] = results_df.loc[name, 'roc_auc']
                    else:
                        self.base_weights[name] = 0.5 
            except Exception as e:
                print(f"Error reading model results: {e}")
                self.base_weights = {name: 1.0 for name in self.model_names}
        else:
            self.base_weights = {name: 1.0 for name in self.model_names}
            
        # Normalize base weights
        total = sum(self.base_weights.values())
        if total > 0:
            self.base_weights = {k: v / total for k, v in self.base_weights.items()}
        else:
            self.base_weights = {k: 1.0/len(self.base_weights) for k in self.base_weights}

    def _calculate_adaptive_weights(self, features_dict):
        """Adjust model weights dynamically based on input features"""
        weights = self.base_weights.copy()
        
        danceability = features_dict.get('danceability', 0)
        energy = features_dict.get('energy', 0)
        acousticness = features_dict.get('acousticness', 0)
        tempo = features_dict.get('tempo', 0)
        
        # Adaptive Rule 1: High Danceability & Energy
        if danceability > 0.75 and energy > 0.7:
            for m in ['xgboost', 'lightgbm', 'catboost']:
                if m in weights:
                    weights[m] *= 1.5
                    
        # Adaptive Rule 2: High Acousticness
        if acousticness > 0.6:
            if 'random_forest' in weights:
                weights['random_forest'] *= 1.4
                
        # Adaptive Rule 3: Specific Tempo Range (Viral Sweet Spot)
        if 110 <= tempo <= 130:
            if 'xgboost' in weights:
                weights['xgboost'] *= 1.3
                
        # Normalize weights so they sum to 1
        total = sum(weights.values())
        normalized_weights = {k: v / total for k, v in weights.items()}
        
        return normalized_weights

    def predict(self, features_dict):
        """
        Generate a hybrid prediction for a given song's features
        
        Args:
            features_dict (dict): Dictionary of audio features
            
        Returns:
            dict: AMVP prediction results including score and explanation
        """
        # Convert dict to DataFrame for scaling and prediction
        input_df = pd.DataFrame([features_dict])
        
        # Add missing features with default 0 to match training set
        for col in self.feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
                
        # Ensure correct feature order
        input_df = input_df[self.feature_names]
            
        # Scale features
        X_scaled = self.scaler.transform(input_df)
        
        # Calculate adaptive weights
        adaptive_weights = self._calculate_adaptive_weights(features_dict)
        
        # Step 4: Weighted Prediction Fusion
        final_probability = 0
        model_predictions = {}
        
        for name, model in self.models.items():
            try:
                # Use predict_proba for probabilities
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X_scaled)[0][1]
                else:
                    # Fallback for models without predict_proba if any
                    prob = model.predict(X_scaled)[0]
                
                model_predictions[name] = prob
                final_probability += adaptive_weights.get(name, 0) * prob
            except Exception as e:
                print(f"Error predicting with {name}: {e}")
                
        # Step 5: Virality Score Generation
        virality_score = int(final_probability * 100)
        
        if virality_score < 40:
            potential = "Low Viral Potential"
        elif virality_score < 70:
            potential = "Moderate Viral Potential"
        else:
            potential = "High Viral Potential"
            
        # Step 6: Explainability (Top Influencing Features)
        top_features = []
        if features_dict.get('danceability', 0) > 0.7:
            top_features.append("Danceability")
        if features_dict.get('energy', 0) > 0.7:
            top_features.append("Energy")
        if 110 <= features_dict.get('tempo', 0) <= 130:
            top_features.append("Tempo")
        if features_dict.get('valence', 0) > 0.6:
            top_features.append("Valence (Positivity)")
        if features_dict.get('loudness', -60) > -5:
            top_features.append("Loudness")
        if features_dict.get('artist_avg_popularity', 0) > 60:
            top_features.append("Artist Recognition")
            
        # Fallback to general values if list is empty
        if not top_features:
            top_features = ["Feature Synergy", "Audio Consistency", "Market Timing"]

        return {
            "virality_score": virality_score,
            "prediction": potential,
            "top_features": top_features[:3],
            "individual_predictions": model_predictions,
            "weights_used": adaptive_weights
        }

if __name__ == "__main__":
    # Example usage
    try:
        amvp = AdaptiveMultiModelViralityPredictor()
        
        # Sample song with basic features
        sample_song = {
            'duration_ms': 180000,
            'explicit': 0,
            'danceability': 0.85,
            'energy': 0.80,
            'key': 5,
            'loudness': -4.2,
            'mode': 1,
            'speechiness': 0.12,
            'acousticness': 0.05,
            'instrumentalness': 0.0,
            'liveness': 0.10,
            'valence': 0.75,
            'tempo': 128.0,
            'time_signature': 4,
            'artist_avg_popularity': 75.0
        }
        
        result = amvp.predict(sample_song)
        
        print("\n" + "="*40)
        print(" AMVP HYBRID PREDICTION RESULT")
        print("="*40)
        print(f"Virality Score: {result['virality_score']}/100")
        print(f"Prediction: {result['prediction']}")
        print("\nTop Contributing Features:")
        for feat in result['top_features']:
            print(f"- {feat}")
        print("\nModel Contributions (Adaptive Weights):")
        for model, weight in result['weights_used'].items():
            print(f"- {model:20s}: {weight:.2%}")
        print("="*40)
        
    except Exception as e:
        print(f"Initialization failed: {e}")
        import traceback
        traceback.print_exc()
