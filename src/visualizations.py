"""
Visualization Script (Enhanced)
Creates comprehensive visualizations for EDA, model results, SHAP
explainability, threshold optimization, and cross-validation results.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import joblib
import json
import argparse
import os

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class VisualizationGenerator:
    """Generate visualizations for the project"""

    def __init__(self, output_dir='visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Feature distributions
    # ------------------------------------------------------------------ #
    def plot_feature_distributions(self, df, features):
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.ravel()
        for idx, feature in enumerate(features[:9]):
            if feature in df.columns:
                axes[idx].hist(df[feature], bins=50, edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'Distribution of {feature}', fontsize=10, fontweight='bold')
                axes[idx].set_xlabel(feature)
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: feature_distributions.png")

    # ------------------------------------------------------------------ #
    #  Correlation matrix
    # ------------------------------------------------------------------ #
    def plot_correlation_matrix(self, df, features):
        plt.figure(figsize=(14, 12))
        numeric_features = [f for f in features if f in df.columns and df[f].dtype in ['float64', 'int64']]
        corr_matrix = df[numeric_features].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm',
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: correlation_matrix.png")

    # ------------------------------------------------------------------ #
    #  Viral vs non-viral comparison
    # ------------------------------------------------------------------ #
    def plot_viral_vs_nonviral(self, df):
        audio_features = ['danceability', 'energy', 'valence', 'loudness',
                          'speechiness', 'acousticness']
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        for idx, feature in enumerate(audio_features):
            if feature in df.columns:
                viral = df[df['is_viral'] == 1][feature]
                non_viral = df[df['is_viral'] == 0][feature]
                axes[idx].hist(non_viral, bins=30, alpha=0.5, label='Non-Viral', color='blue')
                axes[idx].hist(viral, bins=30, alpha=0.5, label='Viral', color='red')
                axes[idx].set_title(f'{feature.capitalize()}', fontsize=12, fontweight='bold')
                axes[idx].set_xlabel(feature)
                axes[idx].set_ylabel('Frequency')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
        plt.suptitle('Viral vs Non-Viral Songs: Audio Features', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'viral_vs_nonviral.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: viral_vs_nonviral.png")

    # ------------------------------------------------------------------ #
    #  Feature importance
    # ------------------------------------------------------------------ #
    def plot_feature_importance(self, model, feature_names, model_name):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]
            plt.figure(figsize=(10, 8))
            plt.title(f'Top 20 Feature Importance - {model_name}', fontsize=14, fontweight='bold')
            plt.barh(range(len(indices)), importances[indices], color='steelblue')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Importance Score')
            plt.gca().invert_yaxis()
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            filename = f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {filename}")
        elif hasattr(model, 'coef_'):
            coef = np.abs(model.coef_[0])
            indices = np.argsort(coef)[::-1][:20]
            plt.figure(figsize=(10, 8))
            plt.title(f'Top 20 Feature Coefficients - {model_name}', fontsize=14, fontweight='bold')
            plt.barh(range(len(indices)), coef[indices], color='coral')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Absolute Coefficient Value')
            plt.gca().invert_yaxis()
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            filename = f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {filename}")

    # ------------------------------------------------------------------ #
    #  Confusion matrices
    # ------------------------------------------------------------------ #
    def plot_confusion_matrices(self, results, y_test):
        n_models = len(results)
        cols = 2
        rows = max((n_models + 1) // 2, 1)
        fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 5))
        axes = np.array(axes).ravel() if n_models > 1 else [axes]
        for idx, (model_name, result) in enumerate(results.items()):
            cm = confusion_matrix(y_test, result['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Not Viral', 'Viral'],
                        yticklabels=['Not Viral', 'Viral'],
                        ax=axes[idx], cbar=True)
            axes[idx].set_title(f'{model_name}\nAccuracy: {result["accuracy"]:.3f}',
                                fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        plt.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: confusion_matrices.png")

    # ------------------------------------------------------------------ #
    #  ROC curves
    # ------------------------------------------------------------------ #
    def plot_roc_curves(self, results, y_test):
        plt.figure(figsize=(10, 8))
        for model_name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: roc_curves.png")

    # ------------------------------------------------------------------ #
    #  Model comparison (bar + radar)
    # ------------------------------------------------------------------ #
    def plot_model_comparison(self, results_df):
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        available_metrics = [m for m in metrics if m in results_df.columns]
        if not available_metrics:
            return

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # Bar plot
        results_df[available_metrics].plot(kind='bar', ax=axes[0], width=0.8)
        axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Model', fontsize=12)
        axes[0].set_ylabel('Score', fontsize=12)
        axes[0].set_xticklabels(results_df.index, rotation=45, ha='right')
        axes[0].legend(loc='lower right')
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim([0, 1])

        # Radar plot
        from math import pi
        N = len(available_metrics)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        ax = plt.subplot(122, polar=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(results_df)))
        for idx, (model_name, row) in enumerate(results_df.iterrows()):
            values = row[available_metrics].tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available_metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=8)
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: model_comparison.png")

    # ------------------------------------------------------------------ #
    #  Popularity distribution
    # ------------------------------------------------------------------ #
    def plot_popularity_distribution(self, df):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(df['popularity'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0].axvline(70, color='red', linestyle='--', linewidth=2, label='Viral Threshold (70)')
        axes[0].set_title('Popularity Score Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Popularity Score')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        df_plot = df[['popularity', 'is_viral']].copy()
        df_plot['Status'] = df_plot['is_viral'].map({0: 'Non-Viral', 1: 'Viral'})
        sns.boxplot(data=df_plot, x='Status', y='popularity', ax=axes[1], palette='Set2')
        axes[1].set_title('Popularity by Viral Status', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Popularity Score')
        axes[1].grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'popularity_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: popularity_distribution.png")

    # ================================================================== #
    #  NEW - Threshold optimization plot
    # ================================================================== #
    def plot_threshold_optimization(self, thresholds_path):
        """Visualize optimal thresholds per model"""
        if not os.path.exists(thresholds_path):
            return
        with open(thresholds_path, 'r') as f:
            thresholds = json.load(f)
        if not thresholds:
            return

        names = list(thresholds.keys())
        values = [thresholds[n] for n in names]

        plt.figure(figsize=(10, max(4, len(names) * 0.5)))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
        bars = plt.barh(names, values, color=colors)
        plt.axvline(0.5, color='red', linestyle='--', linewidth=1.5, label='Default (0.5)')
        for bar, v in zip(bars, values):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{v:.3f}', va='center', fontsize=10)
        plt.xlabel('Optimal Threshold', fontsize=12)
        plt.title('Optimal Classification Threshold per Model (max F1)',
                  fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'threshold_optimization.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: threshold_optimization.png")

    # ================================================================== #
    #  NEW - Cross-validation results
    # ================================================================== #
    def plot_cv_results(self, cv_path):
        """Visualize cross-validation AUC scores per model"""
        if not os.path.exists(cv_path):
            return
        with open(cv_path, 'r') as f:
            cv_results = json.load(f)
        if not cv_results:
            return

        names = []
        means = []
        stds = []
        all_scores = []
        for name, data in cv_results.items():
            names.append(name)
            means.append(data['mean_auc'])
            stds.append(data['std_auc'])
            all_scores.append(data['scores'])

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Box plot
        axes[0].boxplot(all_scores, labels=names, vert=True, patch_artist=True,
                        boxprops=dict(facecolor='lightblue'))
        axes[0].set_title('Cross-Validation ROC-AUC Distribution',
                          fontsize=14, fontweight='bold')
        axes[0].set_ylabel('ROC-AUC')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)

        # Bar chart with error bars
        x = np.arange(len(names))
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        axes[1].bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=45, ha='right')
        axes[1].set_ylabel('Mean ROC-AUC')
        axes[1].set_title('Mean CV ROC-AUC (+/- std)',
                          fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        for i, (m, s) in enumerate(zip(means, stds)):
            axes[1].text(i, m + s + 0.005, f'{m:.3f}', ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cv_results.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: cv_results.png")

    # ================================================================== #
    #  NEW - Regression results plot
    # ================================================================== #
    def plot_regression_results(self, reg_results_path):
        """Visualize regression model comparison"""
        if not os.path.exists(reg_results_path):
            return
        reg_df = pd.read_csv(reg_results_path, index_col=0)
        if reg_df.empty:
            return

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for i, metric in enumerate(['rmse', 'mae', 'r2']):
            if metric in reg_df.columns:
                colors = plt.cm.Set3(np.linspace(0, 1, len(reg_df)))
                reg_df[metric].plot(kind='bar', ax=axes[i], color=colors, edgecolor='black')
                axes[i].set_title(metric.upper(), fontsize=14, fontweight='bold')
                axes[i].set_ylabel(metric.upper())
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(axis='y', alpha=0.3)
        plt.suptitle('Regression Model Comparison (Popularity Prediction)',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'regression_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: regression_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='Generate visualizations')
    parser.add_argument('--data', type=str, required=True, help='Path to data CSV')
    parser.add_argument('--results', type=str, help='Path to model results CSV')
    parser.add_argument('--models_dir', type=str, help='Directory containing saved models')
    parser.add_argument('--output', type=str, default='visualizations/', help='Output directory')
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)

    viz = VisualizationGenerator(args.output)

    # --- EDA visualizations ---
    print("\nGenerating EDA visualizations...")
    audio_features = ['danceability', 'energy', 'valence', 'loudness',
                      'speechiness', 'acousticness', 'tempo', 'duration_min']
    if 'popularity' in df.columns:
        viz.plot_popularity_distribution(df)
    viz.plot_feature_distributions(df, audio_features)
    viz.plot_correlation_matrix(df, audio_features)
    if 'is_viral' in df.columns:
        viz.plot_viral_vs_nonviral(df)

    # --- Model performance visualizations ---
    if args.results and os.path.exists(args.results):
        print("\nGenerating model performance visualizations...")
        results_df = pd.read_csv(args.results, index_col=0)
        viz.plot_model_comparison(results_df)

    # --- NEW: Threshold optimization ---
    if args.models_dir:
        thresh_path = os.path.join(args.models_dir, 'optimal_thresholds.json')
        viz.plot_threshold_optimization(thresh_path)

    # --- NEW: Cross-validation results ---
    if args.models_dir:
        cv_path = os.path.join(args.models_dir, 'cv_results.json')
        viz.plot_cv_results(cv_path)

    # --- NEW: Regression results ---
    if args.models_dir:
        reg_path = os.path.join(args.models_dir, 'regression_results.csv')
        viz.plot_regression_results(reg_path)

    print("\n" + "="*50)
    print("Visualization generation completed!")
    print("="*50)


if __name__ == '__main__':
    main()
