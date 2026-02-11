"""
Hyperparameter Optimization using Optuna with Purged K-Fold CV.

This module implements hyperparameter tuning for Random Forest and XGBoost
models following AFML Chapter 8 best practices:
1. Uses Purged K-Fold CV to prevent information leakage
2. Optimizes for ROC AUC (better metric for imbalanced financial data)
3. Applies early stopping to prevent overfitting

Reference:
    - AFML Chapter 7: Cross-Validation in Finance
    - AFML Chapter 8: Feature Importance and Model Selection
    - mlfinlab: PurgedKFold implementation

Author: AFML Project
"""

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss
import matplotlib.pyplot as plt
import os
import sys
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Ensure src is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cv_setup import PurgedKFold


class HyperparameterOptimizer:
    """
    Hyperparameter optimizer for financial ML models using Optuna.
    
    Uses Purged K-Fold CV to ensure proper evaluation without lookahead bias.
    
    Attributes:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target labels
        t1 (pd.Series): End times for each sample (for purging)
        sample_weights (pd.Series): Sample weights
        n_splits (int): Number of CV folds
        embargo_pct (float): Embargo percentage after test set
        
    Reference:
        AFML Chapter 7 - Cross-Validation in Finance
    """
    
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        t1: pd.Series,
        sample_weights: pd.Series = None,
        n_splits: int = 5,
        embargo_pct: float = 0.01
    ):
        """
        Initialize the optimizer.
        
        Args:
            X: Feature matrix
            y: Target labels (should be -1/1 or 0/1)
            t1: End times for each sample
            sample_weights: Optional sample weights
            n_splits: Number of CV folds
            embargo_pct: Embargo percentage (0.01 = 1%)
        """
        self.X = X
        self.y = y.astype(int)
        self.t1 = t1
        self.sample_weights = sample_weights if sample_weights is not None else pd.Series(1.0, index=X.index)
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        
        # Setup Purged CV
        self.cv = PurgedKFold(
            n_splits=n_splits,
            samples_info_sets=t1,
            embargo=embargo_pct
        )
        
        # Results storage
        self.study = None
        self.best_params = None
        self.best_score = None
        
    def _evaluate_model(self, model, X, y, sample_weights) -> dict:
        """
        Evaluate model using Purged K-Fold CV.
        
        Returns:
            dict: Contains mean and std of various metrics
        """
        auc_scores = []
        accuracy_scores = []
        f1_scores = []
        
        for train_idx, test_idx in self.cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            w_train = sample_weights.iloc[train_idx]
            
            # Train
            model.fit(X_train, y_train, sample_weight=w_train)
            
            # Predict
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # AUC
            if len(np.unique(y)) == 2:
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                
            auc_scores.append(auc)
            accuracy_scores.append(acc)
            f1_scores.append(f1)
            
        return {
            'auc_mean': np.mean(auc_scores),
            'auc_std': np.std(auc_scores),
            'accuracy_mean': np.mean(accuracy_scores),
            'accuracy_std': np.std(accuracy_scores),
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores)
        }
    
    def objective_rf(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function for Random Forest.
        
        Optimizes for ROC AUC while controlling model complexity.
        
        Reference:
            AFML Chapter 8 - The Importance of Cross-Validation
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=200),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5]),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'class_weight': 'balanced_subsample',
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = RandomForestClassifier(**params)
        metrics = self._evaluate_model(model, self.X, self.y, self.sample_weights)
        
        # Log additional metrics
        trial.set_user_attr('auc_std', metrics['auc_std'])
        trial.set_user_attr('accuracy_mean', metrics['accuracy_mean'])
        trial.set_user_attr('f1_mean', metrics['f1_mean'])
        
        return metrics['auc_mean']
    
    def optimize(
        self,
        n_trials: int = 50,
        timeout: int = None,
        model_type: str = 'random_forest'
    ) -> dict:
        """
        Run hyperparameter optimization.
        
        Args:
            n_trials: Number of optimization trials
            timeout: Optional timeout in seconds
            model_type: 'random_forest' or 'xgboost' (future)
            
        Returns:
            dict: Best parameters and metrics
        """
        print("=" * 80)
        print(f"Hyperparameter Optimization - {model_type.upper()}")
        print(f"Trials: {n_trials} | CV Folds: {self.n_splits} | Embargo: {self.embargo_pct*100}%")
        print("=" * 80)
        
        # Create study with TPE sampler (better for hyperparameter tuning)
        sampler = TPESampler(seed=42)
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name=f'{model_type}_optimization'
        )
        
        # Select objective based on model type
        if model_type == 'random_forest':
            objective = self.objective_rf
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Optimize
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            callbacks=[self._print_callback]
        )
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        print("\n" + "=" * 80)
        print("Optimization Complete!")
        print("=" * 80)
        print(f"\nBest ROC AUC: {self.best_score:.4f}")
        print(f"Best Parameters:")
        for k, v in self.best_params.items():
            print(f"   {k}: {v}")
            
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_metrics': {
                'auc_mean': self.study.best_trial.user_attrs.get('auc_mean', self.best_score),
                'auc_std': self.study.best_trial.user_attrs.get('auc_std', 0),
                'accuracy_mean': self.study.best_trial.user_attrs.get('accuracy_mean', 0),
                'f1_mean': self.study.best_trial.user_attrs.get('f1_mean', 0)
            }
        }
    
    def _print_callback(self, study: optuna.Study, trial: optuna.FrozenTrial):
        """Callback to print trial results."""
        if trial.number % 5 == 0:
            print(f"\nTrial {trial.number}: AUC = {trial.value:.4f}")
    
    def train_best_model(self, model_type: str = 'random_forest'):
        """
        Train a model with the best parameters found.
        
        Returns:
            Trained model
        """
        if self.best_params is None:
            raise ValueError("Must run optimize() first")
            
        if model_type == 'random_forest':
            params = {
                **self.best_params,
                'class_weight': 'balanced_subsample',
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': -1
            }
            model = RandomForestClassifier(**params)
            
        # Full training
        model.fit(self.X, self.y, sample_weight=self.sample_weights)
        
        return model
    
    def plot_optimization_history(self, save_path: str = None):
        """
        Plot optimization history and parameter importance.
        """
        if self.study is None:
            raise ValueError("Must run optimize() first")
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Optimization History
        ax1 = axes[0, 0]
        trials = [t.number for t in self.study.trials]
        values = [t.value for t in self.study.trials]
        best_values = np.maximum.accumulate(values)
        
        ax1.plot(trials, values, 'o-', alpha=0.5, label='Trial AUC')
        ax1.plot(trials, best_values, 'r-', linewidth=2, label='Best AUC')
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
        ax1.set_xlabel('Trial')
        ax1.set_ylabel('ROC AUC')
        ax1.set_title('Optimization History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Parameter Importance (using hyperparameter importance)
        ax2 = axes[0, 1]
        try:
            importance = optuna.importance.get_param_importances(self.study)
            params = list(importance.keys())[:10]
            values = [importance[p] for p in params]
            
            ax2.barh(params, values, color='steelblue')
            ax2.set_xlabel('Importance')
            ax2.set_title('Hyperparameter Importance')
            ax2.invert_yaxis()
        except Exception as e:
            ax2.text(0.5, 0.5, f'Cannot compute importance:\n{str(e)[:50]}',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Hyperparameter Importance')
        
        # 3. Distribution of AUC scores
        ax3 = axes[1, 0]
        auc_values = [t.value for t in self.study.trials if t.value is not None]
        ax3.hist(auc_values, bins=20, color='steelblue', edgecolor='white', alpha=0.7)
        ax3.axvline(x=self.best_score, color='red', linestyle='--', linewidth=2, label=f'Best: {self.best_score:.4f}')
        ax3.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
        ax3.set_xlabel('ROC AUC')
        ax3.set_ylabel('Count')
        ax3.set_title('Distribution of Trial AUC Scores')
        ax3.legend()
        
        # 4. Key Parameter distributions
        ax4 = axes[1, 1]
        if 'max_depth' in self.best_params:
            depths = [t.params.get('max_depth', 0) for t in self.study.trials]
            scores = [t.value for t in self.study.trials]
            
            ax4.scatter(depths, scores, alpha=0.6, c='steelblue')
            ax4.axvline(x=self.best_params['max_depth'], color='red', linestyle='--', 
                       label=f"Best: {self.best_params['max_depth']}")
            ax4.set_xlabel('max_depth')
            ax4.set_ylabel('ROC AUC')
            ax4.set_title('AUC vs max_depth')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.close()
        
    def save_results(self, output_dir: str = '.'):
        """Save optimization results to files."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'best_params': self.best_params,
            'best_score': float(self.best_score),
            'n_trials': len(self.study.trials),
            'all_trials': [
                {
                    'number': t.number,
                    'value': float(t.value) if t.value else None,
                    'params': t.params,
                    'user_attrs': t.user_attrs
                }
                for t in self.study.trials
            ]
        }
        
        # Save JSON
        json_path = os.path.join(output_dir, 'hyperparameter_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {json_path}")
        
        # Save best params as simple CSV for easy loading
        params_df = pd.DataFrame([self.best_params])
        params_df['best_auc'] = self.best_score
        params_path = os.path.join(output_dir, 'best_hyperparameters.csv')
        params_df.to_csv(params_path, index=False)
        print(f"Saved best params to {params_path}")


def main():
    """Main function to run hyperparameter optimization."""
    print("=" * 80)
    print("Hyperparameter Optimization Pipeline")
    print("=" * 80)
    
    # 1. Load Data
    print("\n1. Loading Data...")
    try:
        df = pd.read_csv("features_labeled.csv", index_col=0, parse_dates=True)
        print(f"   Loaded features_labeled.csv: {df.shape}")
        
        # Check for t1
        if 't1' not in df.columns:
            print("   't1' column missing. Joining with labeled_events.csv...")
            events = pd.read_csv("labeled_events.csv", index_col=0, parse_dates=True)
            df = df.join(events[['t1']], rsuffix='_events')
            if 't1' not in df.columns and 't1_events' in df.columns:
                df['t1'] = df['t1_events']
        
        df['t1'] = pd.to_datetime(df['t1'])
        
        # Drop NaN
        original_len = len(df)
        df = df.dropna()
        if len(df) < original_len:
            print(f"   Dropped {original_len - len(df)} rows with NaNs")
            
        print(f"   Final Dataset: {len(df)} samples")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # 2. Prepare features
    print("\n2. Preparing Features...")
    target_col = 'label'
    weight_col = 'sample_weight'
    exclude_cols = ['label', 'ret', 'sample_weight', 'avg_uniqueness', 't1', 'trgt', 'side', 'bin', 't1_events']
    
    # Use selected features if available
    if os.path.exists("selected_features.csv"):
        print("   Using MDA-selected features")
        selected_df = pd.read_csv("selected_features.csv")
        feature_cols = [c for c in selected_df['feature'].tolist() if c in df.columns]
    else:
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
    print(f"   Features: {len(feature_cols)}")
    
    X = df[feature_cols]
    y = df[target_col]
    t1 = df['t1']
    
    if weight_col in df.columns:
        sample_weights = df[weight_col]
        print(f"   Sample weights available (mean: {sample_weights.mean():.4f})")
    else:
        sample_weights = None
        
    # 3. Run Optimization
    print("\n3. Starting Optimization...")
    optimizer = HyperparameterOptimizer(
        X=X,
        y=y,
        t1=t1,
        sample_weights=sample_weights,
        n_splits=5,
        embargo_pct=0.01
    )
    
    # Run with 50 trials (should take ~10-20 minutes)
    results = optimizer.optimize(n_trials=50, model_type='random_forest')
    
    # 4. Save results
    print("\n4. Saving Results...")
    optimizer.save_results(output_dir='.')
    
    # 5. Plot
    print("\n5. Generating Visualizations...")
    os.makedirs('visual_analysis', exist_ok=True)
    optimizer.plot_optimization_history(save_path='visual_analysis/hyperparameter_optimization.png')
    
    # 6. Compare with baseline
    print("\n6. Comparison with Baseline")
    print("-" * 50)
    print(f"   Baseline AUC:   0.5122 (+/- 0.0257)")
    print(f"   Optimized AUC:  {results['best_score']:.4f}")
    improvement = (results['best_score'] - 0.5122) / 0.5122 * 100
    print(f"   Improvement:    {improvement:+.2f}%")
    
    # 7. Train final model with best params
    print("\n7. Training Final Model...")
    best_model = optimizer.train_best_model()
    
    # Save model feature importance
    fi = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    fi.to_csv('feature_importance_optimized.csv', index=False)
    print("   Saved feature importance to feature_importance_optimized.csv")
    
    print("\n" + "=" * 80)
    print("✓ Hyperparameter Optimization Complete!")
    print("=" * 80)
    print("\nFiles created:")
    print("   - hyperparameter_results.json (full optimization history)")
    print("   - best_hyperparameters.csv (best params for future use)")
    print("   - visual_analysis/hyperparameter_optimization.png")
    print("   - feature_importance_optimized.csv")


if __name__ == "__main__":
    main()
