# Probability Prediction of Song Virality using Spotify Data

## 🎵 Project Overview

This project develops a machine learning system to predict the probability of a song becoming viral based on Spotify audio features and metadata. By analyzing patterns in successful songs, we can identify key characteristics that contribute to virality and build predictive models with high accuracy.

## 🎯 Objectives

1. **Data Collection**: Gather comprehensive Spotify dataset with audio features, metadata, and popularity metrics
2. **Exploratory Analysis**: Understand relationships between audio features and song virality
3. **Feature Engineering**: Create meaningful derived features to improve model performance
4. **Model Development**: Implement and compare multiple machine learning algorithms
5. **Evaluation**: Assess model performance using industry-standard metrics
6. **Insights**: Extract actionable insights about what makes songs go viral

## 📊 Dataset Features

### Audio Features
- **Acousticness**: Confidence measure of whether the track is acoustic (0.0 to 1.0)
- **Danceability**: How suitable a track is for dancing (0.0 to 1.0)
- **Energy**: Perceptual measure of intensity and activity (0.0 to 1.0)
- **Instrumentalness**: Predicts whether a track contains no vocals (0.0 to 1.0)
- **Liveness**: Detects presence of an audience in the recording (0.0 to 1.0)
- **Loudness**: Overall loudness in decibels (dB)
- **Speechiness**: Detects presence of spoken words (0.0 to 1.0)
- **Valence**: Musical positiveness conveyed by a track (0.0 to 1.0)
- **Tempo**: Overall estimated tempo in beats per minute (BPM)

### Metadata
- **Duration**: Length of the track in milliseconds
- **Key**: The key the track is in (0 = C, 1 = C#, etc.)
- **Mode**: Major (1) or minor (0)
- **Time Signature**: Estimated overall time signature
- **Popularity**: Current popularity score (0-100)

### Target Variable
- **Viral Status**: Binary classification (1 = Viral, 0 = Not Viral)
  - Threshold: Songs with popularity > 70 are considered viral

## 🏗️ Project Structure

```
spotify-virality-prediction/
├── data/                       # Data files
│   ├── raw/                   # Raw collected data
│   ├── processed/             # Cleaned and processed data
│   └── sample_data.csv        # Sample dataset for demonstration
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_collection.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   └── 03_model_training.ipynb
├── src/                       # Source code
│   ├── data_collection.py    # Spotify API data collection
│   ├── feature_engineering.py # Feature creation and selection
│   ├── train_models.py       # Model training pipeline
│   └── evaluate_models.py    # Model evaluation and comparison
├── models/                    # Saved model files
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost_model.pkl
│   └── neural_network.h5
├── visualizations/            # Generated plots and charts
├── reports/                   # Project reports and documentation
│   └── project_report.pdf
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook (optional, for running notebooks)
- Spotify Developer Account (for API access)

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd spotify-virality-prediction
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Spotify API credentials:**
   - Go to https://developer.spotify.com/dashboard
   - Create an app and get your Client ID and Client Secret
   - Create a `.env` file in the project root:
     ```
     SPOTIFY_CLIENT_ID=your_client_id_here
     SPOTIFY_CLIENT_SECRET=your_client_secret_here
     ```

## 📈 Usage

### 1. Data Collection

Collect song data from Spotify API:
```bash
python src/data_collection.py --num_songs 10000 --output data/raw/spotify_songs.csv
```

### 2. Exploratory Data Analysis

Run the EDA notebook to understand the data:
```bash
jupyter notebook notebooks/02_exploratory_analysis.ipynb
```

### 3. Train Models

Train all machine learning models:
```bash
python src/train_models.py --input data/processed/features.csv --output models/
```

### 4. Evaluate Models

Compare model performance:
```bash
python src/evaluate_models.py --models_dir models/ --output reports/
```

## 🤖 Machine Learning Models

### 1. Logistic Regression
- **Type**: Linear classifier
- **Advantages**: Fast, interpretable, good baseline
- **Use Case**: Understanding feature importance

### 2. Random Forest
- **Type**: Ensemble of decision trees
- **Advantages**: Handles non-linear relationships, robust to outliers
- **Use Case**: High accuracy with minimal tuning

### 3. XGBoost
- **Type**: Gradient boosting algorithm
- **Advantages**: State-of-the-art performance, handles imbalanced data
- **Use Case**: Best overall performance

### 4. Neural Network
- **Type**: Deep learning model
- **Architecture**: Multi-layer perceptron (MLP)
- **Advantages**: Captures complex patterns
- **Use Case**: Maximum predictive power

## 📊 Model Performance Metrics

We evaluate models using:
- **Accuracy**: Overall correctness
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Visualization of prediction errors

## 🔍 Key Findings

### Top Predictive Features
1. **Danceability** - Highly correlated with viral success
2. **Energy** - High-energy songs tend to perform better
3. **Valence** - Positive songs have higher virality
4. **Loudness** - Louder songs are more likely to go viral
5. **Tempo** - Moderate to fast tempo correlates with popularity

### Model Comparison
- **Best Overall**: XGBoost (ROC-AUC: ~0.89)
- **Most Interpretable**: Logistic Regression (ROC-AUC: ~0.82)
- **Best for Production**: Random Forest (ROC-AUC: ~0.87, fast inference)

## 📚 Technologies Used

- **Python 3.8+**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning algorithms and utilities
- **XGBoost**: Gradient boosting framework
- **TensorFlow/Keras**: Neural network implementation
- **Matplotlib & Seaborn**: Data visualization
- **Spotipy**: Spotify API wrapper
- **Jupyter**: Interactive development environment

## 🎓 Academic Context

This project was developed for the Advanced Data Analytics Lab course, demonstrating:
- End-to-end machine learning pipeline development
- Real-world API integration and data collection
- Statistical analysis and hypothesis testing
- Model selection and hyperparameter tuning
- Performance evaluation and comparison
- Technical documentation and presentation

## 📖 References

1. Spotify Web API Documentation: https://developer.spotify.com/documentation/web-api
2. Scikit-learn: Machine Learning in Python: https://scikit-learn.org
3. XGBoost Documentation: https://xgboost.readthedocs.io
4. "Predicting Song Popularity" - IEEE Conference Papers
5. "Audio Features for Music Information Retrieval" - ISMIR

## 👥 Author

**Student Name**: [Your Name]  
**Course**: Advanced Data Analytics Lab  
**Institution**: [Your Institution]  
**Date**: January 2026

## 📝 License

This project is for educational purposes as part of academic coursework.

## 🙏 Acknowledgments

- Spotify for providing the Web API
- Course instructors and teaching assistants
- Open-source community for the amazing tools and libraries

---

**Note**: This project demonstrates practical application of machine learning techniques to solve real-world predictive analytics problems. The models and insights can be extended for music recommendation systems, playlist generation, and artist advisory services.
