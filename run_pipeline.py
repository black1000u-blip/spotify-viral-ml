"""
Complete Pipeline Script
Runs the entire project pipeline from data collection to model evaluation
"""

import os
import sys
import subprocess
import argparse


def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "="*70)
    print(f"STEP: {description}")
    print("="*70)
    print(f"Running: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error in: {description}")
        print("Pipeline stopped.")
        sys.exit(1)
    
    print(f"\n✅ Completed: {description}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Run the complete Spotify Virality Prediction pipeline')
    parser.add_argument('--num-songs', type=int, default=5000,
                        help='Number of songs to collect (default: 5000)')
    parser.add_argument('--skip-collection', action='store_true',
                        help='Skip data collection (use existing data)')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip model training')
    parser.add_argument('--skip-viz', action='store_true',
                        help='Skip visualization generation')
    parser.add_argument('--no-optuna', action='store_true',
                        help='Skip Optuna hyperparameter tuning')
    parser.add_argument('--no-mlflow', action='store_true',
                        help='Skip MLflow experiment tracking')
    parser.add_argument('--no-shap', action='store_true',
                        help='Skip SHAP explainability analysis')

    args = parser.parse_args()

    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    raw_data = os.path.join(data_dir, 'raw', 'spotify_songs.csv')
    processed_data = os.path.join(data_dir, 'processed', 'features.csv')
    models_dir = os.path.join(base_dir, 'models')
    viz_dir = os.path.join(base_dir, 'visualizations')
    src_dir = os.path.join(base_dir, 'src')

    print("\n" + "="*70)
    print("SPOTIFY SONG VIRALITY PREDICTION - COMPLETE PIPELINE (ENHANCED)")
    print("="*70)

    # Step 1: Data Collection
    if not args.skip_collection:
        cmd = f'python "{os.path.join(src_dir, "data_collection.py")}" --num_songs {args.num_songs} --output "{raw_data}"'
        run_command(cmd, "Data Collection from Spotify")
    else:
        print("\n  Skipping data collection (using existing data)")

    # Step 2: Feature Engineering (now includes temporal, artist, polynomial, target encoding)
    cmd = f'python "{os.path.join(src_dir, "feature_engineering.py")}" --input "{raw_data}" --output "{processed_data}"'
    run_command(cmd, "Feature Engineering (Enhanced)")

    # Step 3: Model Training (all models + Optuna + SHAP + calibration + regression)
    if not args.skip_training:
        train_cmd = f'python "{os.path.join(src_dir, "train_models.py")}" --input "{processed_data}" --output "{models_dir}"'
        if args.no_optuna:
            train_cmd += ' --no-optuna'
        if args.no_mlflow:
            train_cmd += ' --no-mlflow'
        if args.no_shap:
            train_cmd += ' --no-shap'
        run_command(train_cmd, "Model Training & Evaluation (All Models)")
    else:
        print("\n  Skipping model training")

    # Step 4: Generate Visualizations (now includes SHAP, threshold, CV, regression plots)
    if not args.skip_viz:
        results_file = os.path.join(models_dir, 'model_results.csv')
        cmd = (f'python "{os.path.join(src_dir, "visualizations.py")}"'
               f' --data "{processed_data}"'
               f' --results "{results_file}"'
               f' --models_dir "{models_dir}"'
               f' --output "{viz_dir}"')
        run_command(cmd, "Visualization Generation (Enhanced)")
    else:
        print("\n  Skipping visualization generation")

    # Final Summary
    print("\n" + "="*70)
    print("  PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\n  Generated Files:")
    print(f"   - Raw Data:        {raw_data}")
    print(f"   - Processed Data:  {processed_data}")
    print(f"   - Models:          {models_dir}")
    print(f"   - Visualizations:  {viz_dir}")
    print(f"   - SHAP plots:      {os.path.join(models_dir, 'shap')}")
    print(f"   - CV results:      {os.path.join(models_dir, 'cv_results.json')}")
    print(f"   - Thresholds:      {os.path.join(models_dir, 'optimal_thresholds.json')}")
    print("\n  Next Steps:")
    print("   1. Check visualizations/ for all plots")
    print("   2. Review model_results.csv and regression_results.csv")
    print("   3. Review models/shap/ for SHAP explainability")
    print("   4. Start MLflow UI:  mlflow ui --port 5000")
    print("   5. Use trained models for predictions")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
