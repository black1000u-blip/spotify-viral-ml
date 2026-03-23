"""
Spotify Data Collection Script
Collects song data with audio features and metadata from Spotify API.
Now integrated with MongoDB for persistent storage.
"""

import os
import sys
import argparse
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm
import json
from pymongo import MongoClient
import urllib.parse

# Load environment variables
load_dotenv()

class SpotifyDataCollector:
    """Collects song data from Spotify API and stores in MongoDB/CSV"""
    
    def __init__(self, mongo_uri=None):
        """Initialize Spotify API client and MongoDB connection"""
        client_id = os.getenv('SPOTIFY_CLIENT_ID')
        client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        
        if not client_id or not client_secret:
            print("Warning: Spotify credentials not found in environment. Using demo mode.")
            self.sp = None
        else:
            client_credentials_manager = SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret
            )
            self.sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
            
        # MongoDB Setup
        self.mongo_uri = mongo_uri or os.getenv('MONGO_URI')
        self.db = None
        if self.mongo_uri:
            try:
                self.client = MongoClient(self.mongo_uri)
                self.db = self.client['spotify_data']
                print(f"✅ Connected to MongoDB: {self.db.name}")
            except Exception as e:
                print(f"❌ MongoDB Connection Error: {e}")
    
    def get_playlist_tracks(self, playlist_id):
        """Get all tracks from a playlist"""
        if not self.sp: return []
        try:
            results = self.sp.playlist_tracks(playlist_id)
            tracks = results['items']
            while results['next']:
                results = self.sp.next(results)
                tracks.extend(results['items'])
            return tracks
        except Exception as e:
            print(f"Error fetching playlist: {e}")
            return []
    
    def get_audio_features(self, track_ids):
        """Get audio features for a list of track IDs"""
        if not self.sp: return []
        try:
            features = []
            for i in range(0, len(track_ids), 100):
                batch = track_ids[i:i+100]
                batch_features = self.sp.audio_features(batch)
                features.extend([f for f in batch_features if f is not None])
                time.sleep(0.1)
            return features
        except Exception as e:
            print(f"Error fetching audio features: {e}")
            return []
    
    def collect_diverse_songs(self, num_songs=1000):
        """Collect diverse songs from various playlists and genres"""
        if not self.sp:
            dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'dataset.csv')
            if os.path.exists(dataset_path):
                print(f"Loading data from {dataset_path}...")
                try:
                    df = pd.read_csv(dataset_path)
                    if len(df) > num_songs:
                        df = df.sample(n=num_songs, random_state=42)
                    tracks = []
                    for _, row in df.iterrows():
                        tracks.append({
                            'id': row['track_id'],
                            'name': row['track_name'],
                            'artists': [{'name': str(row['artists'])}],
                            'popularity': row['popularity'],
                            'duration_ms': row['duration_ms'],
                            'explicit': row.get('explicit', False),
                            '_audio_features': {
                                'id': row['track_id'],
                                'danceability': row['danceability'],
                                'energy': row['energy'],
                                'key': row['key'],
                                'loudness': row['loudness'],
                                'mode': row['mode'],
                                'speechiness': row['speechiness'],
                                'acousticness': row['acousticness'],
                                'instrumentalness': row['instrumentalness'],
                                'liveness': row['liveness'],
                                'valence': row['valence'],
                                'tempo': row['tempo'],
                                'time_signature': row['time_signature'],
                            }
                        })
                    return tracks
                except Exception as e:
                    print(f"Failed to load dataset: {e}")
                    print("Falling back to generating sample data with noise...")
                    return self._generate_sample_data(num_songs)
            else:
                print("Generating sample data with noise...")
                return self._generate_sample_data(num_songs)
            
        playlists = [
            '37i9dQZEVXbMDoHDwVN2tF', '37i9dQZEVXbLRQDuF5jeBp', '37i9dQZEVXbNG2KDcFcKOF', 
            '37i9dQZF1DXcBWIGoYBM5M', '37i9dQZF1DX0XUsuxWHRQd', '37i9dQZF1DX4dyzvuaRJ0n'
        ]
        
        all_tracks = []
        track_ids_set = set()
        
        for playlist_id in tqdm(playlists, desc="Fetching playlists"):
            tracks = self.get_playlist_tracks(playlist_id)
            for item in tracks:
                if item['track'] and item['track']['id']:
                    tid = item['track']['id']
                    if tid not in track_ids_set:
                        track_ids_set.add(tid)
                        all_tracks.append(item['track'])
                if len(all_tracks) >= num_songs: break
            if len(all_tracks) >= num_songs: break
            time.sleep(0.2)
            
        return all_tracks[:num_songs]
    
    def _generate_sample_data(self, num_songs):
        """Generate realistic sample data for demonstration"""
        np.random.seed(42)
        sample_tracks = []
        artists = ['Artist A', 'Artist B', 'Artist C', 'Artist D', 'Artist E']
        
        for i in range(num_songs):
            # 15% are viral baseline
            is_viral_candidate = (i % 7 == 0)
            pop = int(np.random.uniform(71, 98)) if is_viral_candidate else int(np.random.beta(1.5, 3) * 100)
            
            track = {
                'id': f'track_{i:06d}',
                'name': f'Song {i+1}',
                'artists': [{'name': np.random.choice(artists)}],
                'popularity': pop,
                'duration_ms': np.random.randint(120000, 300000),
                'explicit': np.random.choice([True, False], p=[0.3, 0.7]),
                'album': {'release_date': f'20{np.random.randint(10, 24)}-{np.random.randint(1, 13):02d}-01'}
            }
            sample_tracks.append(track)
        return sample_tracks
    
    def extract_features(self, tracks):
        """Extract features and add noise to audio features for realistic training"""
        print("\nExtracting features...")
        track_ids = [track['id'] for track in tracks]
        
        if self.sp:
            audio_raw = self.get_audio_features(track_ids)
            audio_features_dict = {af['id']: af for af in audio_raw if af}
        else:
            if tracks and '_audio_features' in tracks[0]:
                audio_features_dict = {t['id']: t['_audio_features'] for t in tracks}
            else:
                audio_features_dict = self._generate_sample_audio_features(tracks)
        
        data = []
        for track in tqdm(tracks, desc="Processing tracks"):
            tid = track['id']
            if tid not in audio_features_dict: continue
            
            af = audio_features_dict[tid]
            row = {
                'track_id': tid,
                'track_name': track['name'],
                'artist_name': track['artists'][0]['name'] if track['artists'] else 'Unknown',
                'popularity': track.get('popularity', 0),
                'duration_ms': track.get('duration_ms', 0),
                'explicit': int(track.get('explicit', False)),
            }
            
            if isinstance(af, dict):
                row.update({k: af.get(k, 0) for k in [
                    'danceability', 'energy', 'key', 'loudness', 'mode', 
                    'speechiness', 'acousticness', 'instrumentalness', 
                    'liveness', 'valence', 'tempo', 'time_signature'
                ]})
            
            row['is_viral'] = 1 if row['popularity'] > 70 else 0
            data.append(row)
            
        return pd.DataFrame(data)
    
    def _generate_sample_audio_features(self, tracks):
        """Generate sample audio features with noise correlated to popularity"""
        np.random.seed(42)
        features_dict = {}
        for track in tracks:
            tid = track['id']
            pop = track['popularity']
            is_viral = pop > 70
            
            # ZERO baseline difference (Pure noise / Overlap)
            noise = np.random.normal(0, 0.60)
            
            # Identical baselines for everyone
            base_dance = 0.65
            base_energy = 0.60
            base_val = 0.45
            
            features_dict[tid] = {
                'id': tid,
                'danceability': np.clip(base_dance + noise, 0, 1),
                'energy': np.clip(base_energy + noise, 0, 1),
                'key': np.random.randint(0, 12),
                'loudness': np.clip((-10.0 if is_viral else -12.0) + (noise * 15), -60, 0),
                'mode': np.random.choice([0, 1]),
                'speechiness': np.clip((0.10 if is_viral else 0.08) + noise, 0, 1),
                'acousticness': np.clip((0.30 if is_viral else 0.35) + noise, 0, 1),
                'instrumentalness': np.random.beta(1, 9),
                'liveness': np.random.beta(2, 8),
                'valence': np.clip(base_val + noise, 0, 1),
                'tempo': np.random.uniform(70, 180),
                'time_signature': 4,
            }
        return features_dict

    def save_to_mongodb(self, df):
        """Save dataframe records to MongoDB"""
        if self.db is not None:
            try:
                collection = self.db['raw_songs']
                # Clear old data for fresh run
                collection.delete_many({})
                records = df.to_dict('records')
                collection.insert_many(records)
                print(f"✅ Successfully saved {len(records)} records to MongoDB (raw_songs)")
            except Exception as e:
                print(f"❌ MongoDB Save Error: {e}")
        else:
            print("⚠️ MongoDB not connected. Skipping DB save.")

def main():
    parser = argparse.ArgumentParser(description='Collect Spotify song data')
    parser.add_argument('--num_songs', type=int, default=1000, help='Number of songs to collect')
    parser.add_argument('--output', type=str, default='data/raw/spotify_songs.csv', help='Output CSV path')
    parser.add_argument('--mongo_uri', type=str, default="mongodb+srv://Dradmin:Mongo%40db%23123@cluster0.qa3itof.mongodb.net/", help='MongoDB Connection String')
    
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    collector = SpotifyDataCollector(mongo_uri=args.mongo_uri)
    tracks = collector.collect_diverse_songs(args.num_songs)
    df = collector.extract_features(tracks)
    
    # Save to CSV and MongoDB
    df.to_csv(args.output, index=False)
    collector.save_to_mongodb(df)
    
    print(f"\nData saved to {args.output}")
    print(f"Viral: {df['is_viral'].sum()} ({df['is_viral'].mean()*100:.1f}%)")

if __name__ == '__main__':
    main()
