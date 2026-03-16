"""
Spotify Data Collection Script
Collects song data with audio features and metadata from Spotify API
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

# Load environment variables
load_dotenv()

class SpotifyDataCollector:
    """Collects song data from Spotify API"""
    
    def __init__(self):
        """Initialize Spotify API client"""
        client_id = os.getenv('SPOTIFY_CLIENT_ID')
        client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        
        if not client_id or not client_secret:
            print("Warning: Spotify credentials not found in environment.")
            print("Using demo mode with sample data generation.")
            self.sp = None
        else:
            client_credentials_manager = SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret
            )
            self.sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    
    def get_playlist_tracks(self, playlist_id):
        """Get all tracks from a playlist"""
        if not self.sp:
            return []
        
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
        if not self.sp:
            return []
        
        try:
            # Spotify API allows max 100 tracks per request
            features = []
            for i in range(0, len(track_ids), 100):
                batch = track_ids[i:i+100]
                batch_features = self.sp.audio_features(batch)
                features.extend([f for f in batch_features if f is not None])
                time.sleep(0.1)  # Rate limiting
            return features
        except Exception as e:
            print(f"Error fetching audio features: {e}")
            return []
    
    def search_tracks_by_genre(self, genre, limit=50):
        """Search for tracks by genre"""
        if not self.sp:
            return []
        
        try:
            results = self.sp.search(q=f'genre:{genre}', type='track', limit=limit)
            return results['tracks']['items']
        except Exception as e:
            print(f"Error searching tracks: {e}")
            return []
    
    def collect_diverse_songs(self, num_songs=1000):
        """Collect diverse songs from various playlists and genres"""
        
        # Popular playlist IDs covering different genres and popularity levels
        playlists = [
            '37i9dQZEVXbMDoHDwVN2tF',  # Global Top 50
            '37i9dQZEVXbLRQDuF5jeBp',  # US Top 50
            '37i9dQZEVXbNG2KDcFcKOF',  # Top Songs 2023
            '37i9dQZF1DXcBWIGoYBM5M',  # Today's Top Hits
            '37i9dQZF1DX0XUsuxWHRQd',  # RapCaviar
            '37i9dQZF1DX4dyzvuaRJ0n',  # mint
            '37i9dQZF1DX4WYpdgoIcn6',  # Chill Hits
            '37i9dQZF1DWXRqgorJj26U',  # Rock Classics
            '37i9dQZF1DX4SBhb3fqCJd',  # Are & Be
            '37i9dQZF1DX1lVhptIYRda',  # Hot Country
        ]
        
        genres = ['pop', 'rock', 'hip-hop', 'electronic', 'indie', 
                  'r-n-b', 'jazz', 'classical', 'country', 'latin']
        
        all_tracks = []
        track_ids_set = set()
        
        print("Collecting songs from Spotify...")
        
        if self.sp:
            # Collect from playlists
            for playlist_id in tqdm(playlists, desc="Fetching playlists"):
                tracks = self.get_playlist_tracks(playlist_id)
                for item in tracks:
                    if item['track'] and item['track']['id']:
                        track_id = item['track']['id']
                        if track_id not in track_ids_set:
                            track_ids_set.add(track_id)
                            all_tracks.append(item['track'])
                    
                    if len(all_tracks) >= num_songs:
                        break
                
                if len(all_tracks) >= num_songs:
                    break
                
                time.sleep(0.2)
            
            # If we need more songs, search by genre
            if len(all_tracks) < num_songs:
                for genre in tqdm(genres, desc="Searching by genre"):
                    tracks = self.search_tracks_by_genre(genre, limit=50)
                    for track in tracks:
                        if track['id'] not in track_ids_set:
                            track_ids_set.add(track['id'])
                            all_tracks.append(track)
                        
                        if len(all_tracks) >= num_songs:
                            break
                    
                    if len(all_tracks) >= num_songs:
                        break
                    
                    time.sleep(0.2)
        else:
            # Generate sample data if no API credentials
            print("Generating sample data...")
            all_tracks = self._generate_sample_data(num_songs)
        
        return all_tracks[:num_songs]
    
    def _generate_sample_data(self, num_songs):
        """Generate realistic sample data for demonstration"""
        np.random.seed(42)
        
        sample_tracks = []
        artists = ['Artist A', 'Artist B', 'Artist C', 'Artist D', 'Artist E']
        
        for i in range(num_songs):
            track = {
                'id': f'track_{i:06d}',
                'name': f'Song {i+1}',
                'artists': [{'name': np.random.choice(artists)}],
                'popularity': int(np.random.beta(2, 5) * 100),
                'duration_ms': np.random.randint(120000, 300000),
                'explicit': np.random.choice([True, False], p=[0.3, 0.7]),
                'album': {'release_date': f'20{np.random.randint(10, 24)}-{np.random.randint(1, 13):02d}-01'}
            }
            sample_tracks.append(track)
        
        return sample_tracks
    
    def extract_features(self, tracks):
        """Extract features from track data"""
        
        print("\nExtracting features...")
        
        track_ids = [track['id'] for track in tracks]
        
        if self.sp:
            audio_features = self.get_audio_features(track_ids)
            audio_features_dict = {af['id']: af for af in audio_features if af}
        else:
            # Generate sample audio features
            audio_features_dict = self._generate_sample_audio_features(track_ids)
        
        data = []
        
        for track in tqdm(tracks, desc="Processing tracks"):
            track_id = track['id']
            
            if track_id not in audio_features_dict:
                continue
            
            af = audio_features_dict[track_id]
            
            # Extract basic track info
            row = {
                'track_id': track_id,
                'track_name': track['name'],
                'artist_name': track['artists'][0]['name'] if track['artists'] else 'Unknown',
                'popularity': track.get('popularity', 0),
                'duration_ms': track.get('duration_ms', 0),
                'explicit': int(track.get('explicit', False)),
            }
            
            # Extract audio features
            if isinstance(af, dict):
                row.update({
                    'danceability': af.get('danceability', 0),
                    'energy': af.get('energy', 0),
                    'key': af.get('key', 0),
                    'loudness': af.get('loudness', 0),
                    'mode': af.get('mode', 0),
                    'speechiness': af.get('speechiness', 0),
                    'acousticness': af.get('acousticness', 0),
                    'instrumentalness': af.get('instrumentalness', 0),
                    'liveness': af.get('liveness', 0),
                    'valence': af.get('valence', 0),
                    'tempo': af.get('tempo', 0),
                    'time_signature': af.get('time_signature', 4),
                })
            
            # Create target variable (viral = popularity > 70)
            row['is_viral'] = 1 if row['popularity'] > 70 else 0
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _generate_sample_audio_features(self, track_ids):
        """Generate realistic sample audio features"""
        np.random.seed(42)
        
        features_dict = {}
        
        for track_id in track_ids:
            features_dict[track_id] = {
                'id': track_id,
                'danceability': np.random.beta(5, 2),
                'energy': np.random.beta(5, 2),
                'key': np.random.randint(0, 12),
                'loudness': np.random.uniform(-15, -3),
                'mode': np.random.choice([0, 1]),
                'speechiness': np.random.beta(1, 9),
                'acousticness': np.random.beta(2, 5),
                'instrumentalness': np.random.beta(1, 9),
                'liveness': np.random.beta(2, 8),
                'valence': np.random.beta(5, 5),
                'tempo': np.random.uniform(60, 200),
                'time_signature': np.random.choice([3, 4, 5], p=[0.05, 0.90, 0.05]),
            }
        
        return features_dict


def main():
    """Main function to collect and save data"""
    
    parser = argparse.ArgumentParser(description='Collect Spotify song data')
    parser.add_argument('--num_songs', type=int, default=5000,
                        help='Number of songs to collect (default: 5000)')
    parser.add_argument('--output', type=str, default='data/raw/spotify_songs.csv',
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize collector
    collector = SpotifyDataCollector()
    
    # Collect songs
    tracks = collector.collect_diverse_songs(args.num_songs)
    
    print(f"\nCollected {len(tracks)} tracks")
    
    # Extract features
    df = collector.extract_features(tracks)
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"\nData saved to {args.output}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"Total songs: {len(df)}")
    print(f"Viral songs (popularity > 70): {df['is_viral'].sum()} ({df['is_viral'].mean()*100:.1f}%)")
    print(f"Non-viral songs: {(1-df['is_viral']).sum()} ({(1-df['is_viral']).mean()*100:.1f}%)")
    print(f"\nPopularity distribution:")
    print(df['popularity'].describe())


if __name__ == '__main__':
    main()
