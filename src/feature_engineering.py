"""
Feature Engineering Script (Enhanced)
Creates derived features and prepares data for machine learning models.
Includes: temporal features, artist-level features, genre encoding,
polynomial interactions, and target encoding.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

# Optional: category_encoders for target encoding
try:
    import category_encoders as ce
    CE_AVAILABLE = True
except ImportError:
    CE_AVAILABLE = False


class FeatureEngineer:
    """Feature engineering for Spotify song data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.target_encoder = None
    
    # ------------------------------------------------------------------ #
    #  Duration features
    # ------------------------------------------------------------------ #
    def create_duration_features(self, df):
        """Create features from song duration"""
        df = df.copy()
        df['duration_min'] = df['duration_ms'] / 60000
        df['duration_category'] = pd.cut(
            df['duration_min'],
            bins=[0, 2, 3, 4, 100],
            labels=['very_short', 'short', 'medium', 'long']
        )
        df['optimal_length'] = ((df['duration_min'] >= 2.5) & 
                                 (df['duration_min'] <= 4)).astype(int)
        return df
    
    # ------------------------------------------------------------------ #
    #  Energy & mood features
    # ------------------------------------------------------------------ #
    def create_energy_mood_features(self, df):
        """Create composite features from energy and valence"""
        df = df.copy()
        df['energy_valence'] = df['energy'] * df['valence']
        df['mood'] = 'neutral'
        df.loc[(df['valence'] > 0.6) & (df['energy'] > 0.6), 'mood'] = 'happy_energetic'
        df.loc[(df['valence'] > 0.6) & (df['energy'] <= 0.6), 'mood'] = 'happy_calm'
        df.loc[(df['valence'] <= 0.4) & (df['energy'] > 0.6), 'mood'] = 'sad_energetic'
        df.loc[(df['valence'] <= 0.4) & (df['energy'] <= 0.4), 'mood'] = 'sad_calm'
        df['danceability_energy'] = df['danceability'] * df['energy']
        return df
    
    # ------------------------------------------------------------------ #
    #  Audio balance features
    # ------------------------------------------------------------------ #
    def create_audio_balance_features(self, df):
        """Create features representing audio characteristic balance"""
        df = df.copy()
        df['vocal_instrumental_balance'] = df['speechiness'] - df['instrumentalness']
        df['acoustic_energy_balance'] = df['acousticness'] - df['energy']
        df['is_live'] = (df['liveness'] > 0.8).astype(int)
        df['highly_danceable'] = (df['danceability'] > 0.7).astype(int)
        return df
    
    # ------------------------------------------------------------------ #
    #  Tempo features
    # ------------------------------------------------------------------ #
    def create_tempo_features(self, df):
        """Create features from tempo"""
        df = df.copy()
        df['tempo_category'] = pd.cut(
            df['tempo'],
            bins=[0, 90, 120, 150, 300],
            labels=['slow', 'moderate', 'fast', 'very_fast']
        )
        df['dance_tempo'] = ((df['tempo'] >= 110) & (df['tempo'] <= 130)).astype(int)
        df['tempo_normalized'] = df['tempo'] / 250.0
        return df
    
    # ------------------------------------------------------------------ #
    #  Loudness features
    # ------------------------------------------------------------------ #
    def create_loudness_features(self, df):
        """Create features from loudness"""
        df = df.copy()
        df['loudness_normalized'] = (df['loudness'] + 60) / 60
        df['is_loud'] = (df['loudness'] > -5).astype(int)
        return df
    
    # ------------------------------------------------------------------ #
    #  Interaction / index features
    # ------------------------------------------------------------------ #
    def create_interaction_features(self, df):
        """Create interaction features between key attributes"""
        df = df.copy()
        df['party_index'] = (df['danceability'] * 0.4 + 
                              df['energy'] * 0.3 + 
                              df['valence'] * 0.3)
        df['chill_index'] = (df['acousticness'] * 0.4 + 
                              (1 - df['energy']) * 0.3 + 
                              df['valence'] * 0.3)
        df['workout_index'] = (df['energy'] * 0.5 + 
                                df['tempo'] / 200 * 0.3 + 
                                df['loudness_normalized'] * 0.2)
        return df
    
    # ------------------------------------------------------------------ #
    #  Popularity features
    # ------------------------------------------------------------------ #
    def create_popularity_features(self, df):
        """Create features related to popularity"""
        df = df.copy()
        df['popularity_bucket'] = pd.cut(
            df['popularity'],
            bins=[0, 30, 50, 70, 100],
            labels=['low', 'medium', 'high', 'viral']
        )
        return df
    
    # ------------------------------------------------------------------ #
    #  NEW - Temporal features
    # ------------------------------------------------------------------ #
    def create_temporal_features(self, df):
        """Create features from release date if available"""
        df = df.copy()
        
        if 'release_date' in df.columns:
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
            df['release_year'] = df['release_date'].dt.year.fillna(0).astype(int)
            df['release_month'] = df['release_date'].dt.month.fillna(0).astype(int)
            df['release_day_of_week'] = df['release_date'].dt.dayofweek.fillna(0).astype(int)
            df['is_weekend_release'] = (df['release_day_of_week'] >= 5).astype(int)
            
            # Season of release
            month = df['release_month']
            df['release_season'] = 'unknown'
            df.loc[month.isin([12, 1, 2]), 'release_season'] = 'winter'
            df.loc[month.isin([3, 4, 5]), 'release_season'] = 'spring'
            df.loc[month.isin([6, 7, 8]), 'release_season'] = 'summer'
            df.loc[month.isin([9, 10, 11]), 'release_season'] = 'fall'
            
            # Song age in years (from 2026)
            df['song_age_years'] = 2026 - df['release_year']
            df['song_age_years'] = df['song_age_years'].clip(lower=0)
        else:
            # Generate synthetic temporal features for demo data
            np.random.seed(42)
            n = len(df)
            df['release_year'] = np.random.randint(2015, 2026, size=n)
            df['release_month'] = np.random.randint(1, 13, size=n)
            df['release_day_of_week'] = np.random.randint(0, 7, size=n)
            df['is_weekend_release'] = (df['release_day_of_week'] >= 5).astype(int)
            df['song_age_years'] = 2026 - df['release_year']
        
        return df
    
    # ------------------------------------------------------------------ #
    #  NEW - Artist-level features
    # ------------------------------------------------------------------ #
    def create_artist_features(self, df):
        """Create artist-level aggregate features"""
        df = df.copy()
        
        if 'artist_name' in df.columns:
            artist_stats = df.groupby('artist_name').agg(
                artist_avg_popularity=('popularity', 'mean'),
                artist_song_count=('popularity', 'count'),
                artist_max_popularity=('popularity', 'max'),
                artist_std_popularity=('popularity', 'std')
            ).reset_index()
            artist_stats['artist_std_popularity'] = artist_stats['artist_std_popularity'].fillna(0)
            df = df.merge(artist_stats, on='artist_name', how='left')
            
            # Has the artist had a viral hit before?
            artist_viral = df.groupby('artist_name')['is_viral'].max().reset_index()
            artist_viral.columns = ['artist_name', 'artist_has_viral_hit']
            df = df.merge(artist_viral, on='artist_name', how='left')
        
        return df
    
    # ------------------------------------------------------------------ #
    #  NEW - Target encoding for categorical variables (key, mode)
    # ------------------------------------------------------------------ #
    def create_target_encoded_features(self, df, target_col='is_viral', fit=True):
        """Target-encode categorical variables like key, mode, time_signature"""
        df = df.copy()
        cat_cols = ['key', 'mode', 'time_signature']
        cat_cols = [c for c in cat_cols if c in df.columns]
        
        if CE_AVAILABLE and cat_cols:
            if fit:
                self.target_encoder = ce.TargetEncoder(cols=cat_cols, smoothing=1.0)
                encoded = self.target_encoder.fit_transform(df[cat_cols], df[target_col])
            else:
                encoded = self.target_encoder.transform(df[cat_cols])
            for col in cat_cols:
                df[f'{col}_target_enc'] = encoded[col]
        else:
            # Fallback: simple mean encoding
            for col in cat_cols:
                if col in df.columns:
                    means = df.groupby(col)[target_col].mean()
                    df[f'{col}_target_enc'] = df[col].map(means)
        
        return df
    
    # ------------------------------------------------------------------ #
    #  NEW - Polynomial interaction features
    # ------------------------------------------------------------------ #
    def create_polynomial_features(self, df):
        """Create polynomial (degree-2) interactions for top predictors"""
        df = df.copy()
        top_features = ['danceability', 'energy', 'valence', 'loudness_normalized']
        existing = [f for f in top_features if f in df.columns]
        
        if len(existing) >= 2:
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            poly_arr = poly.fit_transform(df[existing])
            poly_names = poly.get_feature_names_out(existing)
            
            for i, name in enumerate(poly_names):
                if ' ' in name:  # interaction terms contain a space
                    clean_name = name.replace(' ', '_x_')
                    df[f'poly_{clean_name}'] = poly_arr[:, i]
        
        return df
    
    # ------------------------------------------------------------------ #
    #  Encode categorical features
    # ------------------------------------------------------------------ #
    def encode_categorical_features(self, df):
        """Encode categorical features"""
        df = df.copy()
        categorical_cols = ['mood', 'duration_category', 'tempo_category', 'popularity_bucket']
        if 'release_season' in df.columns:
            categorical_cols.append('release_season')
        
        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
        
        return df
    
    # ------------------------------------------------------------------ #
    #  Select final features
    # ------------------------------------------------------------------ #
    def select_model_features(self, df):
        """Select final features for modeling"""
        
        audio_features = [
            'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence',
            'tempo', 'duration_min', 'explicit'
        ]
        engineered_features = [
            'energy_valence', 'danceability_energy', 'vocal_instrumental_balance',
            'acoustic_energy_balance', 'is_live', 'highly_danceable',
            'dance_tempo', 'loudness_normalized', 'is_loud',
            'party_index', 'chill_index', 'workout_index', 'optimal_length',
            'tempo_normalized'
        ]
        temporal_features = [
            'release_year', 'release_month', 'release_day_of_week',
            'is_weekend_release', 'song_age_years'
        ]
        artist_features = [
            'artist_avg_popularity', 'artist_song_count',
            'artist_max_popularity', 'artist_std_popularity',
            'artist_has_viral_hit'
        ]
        target_encoded = [col for col in df.columns if col.endswith('_target_enc')]
        poly_features = [col for col in df.columns if col.startswith('poly_')]
        categorical_encoded = [col for col in df.columns if any(
            prefix in col for prefix in ['mood_', 'duration_category_', 
                                          'tempo_category_', 'popularity_bucket_',
                                          'release_season_']
        )]
        
        feature_cols = (audio_features + engineered_features + temporal_features +
                        artist_features + target_encoded + poly_features +
                        categorical_encoded)
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        return feature_cols
    
    # ------------------------------------------------------------------ #
    #  Master transform
    # ------------------------------------------------------------------ #
    def transform(self, df, fit=True):
        """Apply all feature engineering transformations"""
        print("Creating engineered features...")
        
        # Original features
        df = self.create_duration_features(df)
        df = self.create_energy_mood_features(df)
        df = self.create_audio_balance_features(df)
        df = self.create_tempo_features(df)
        df = self.create_loudness_features(df)
        df = self.create_interaction_features(df)
        df = self.create_popularity_features(df)
        
        # NEW enhanced features
        df = self.create_temporal_features(df)
        df = self.create_artist_features(df)
        df = self.create_target_encoded_features(df, fit=fit)
        df = self.create_polynomial_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Select features for modeling
        feature_cols = self.select_model_features(df)
        print(f"Total features created: {len(feature_cols)}")
        
        return df, feature_cols


def main():
    """Main function for feature engineering"""
    
    parser = argparse.ArgumentParser(description='Engineer features from Spotify data')
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file path')
    parser.add_argument('--output', type=str, default='data/processed/features.csv',
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows")
    
    engineer = FeatureEngineer()
    df_transformed, feature_cols = engineer.transform(df)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_transformed.to_csv(args.output, index=False)
    print(f"\nData saved to {args.output}")
    
    feature_list_path = args.output.replace('.csv', '_features.txt')
    with open(feature_list_path, 'w') as f:
        f.write('\n'.join(feature_cols))
    print(f"Feature list saved to {feature_list_path}")
    
    print(f"\nShape: {df_transformed.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"\nFeature groups:")
    print(f"  Audio:          {sum(1 for c in feature_cols if c in ['danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_min','explicit'])}")
    print(f"  Engineered:     {sum(1 for c in feature_cols if c in ['energy_valence','danceability_energy','vocal_instrumental_balance','acoustic_energy_balance','is_live','highly_danceable','dance_tempo','loudness_normalized','is_loud','party_index','chill_index','workout_index','optimal_length','tempo_normalized'])}")
    print(f"  Temporal:       {sum(1 for c in feature_cols if c in ['release_year','release_month','release_day_of_week','is_weekend_release','song_age_years'])}")
    print(f"  Artist:         {sum(1 for c in feature_cols if 'artist_' in c)}")
    print(f"  Target-encoded: {sum(1 for c in feature_cols if '_target_enc' in c)}")
    print(f"  Polynomial:     {sum(1 for c in feature_cols if c.startswith('poly_'))}")
    print(f"  Categorical:    {sum(1 for c in feature_cols if any(p in c for p in ['mood_','duration_category_','tempo_category_','popularity_bucket_','release_season_']))}")


if __name__ == '__main__':
    main()
