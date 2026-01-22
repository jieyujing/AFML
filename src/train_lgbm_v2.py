import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss
import matplotlib.pyplot as plt
import os
import sys
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Ensure src is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cv_setup import PurgedKFold

class LightGBMOptimizer:
    """
    Hyperparameter optimizer for LightGBM using Optuna and Purged CV.
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
        self.X = X
        self.y = y.astype(int)
        self.t1 = t1
        self.sample_weights = sample_weights if sample_weights is not None else pd.Series(1.0, index=X.index)
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        
        self.cv = PurgedKFold(
            n_splits=n_splits,
            samples_info_sets=t1,
            embargo=embargo_pct
        )
        
        self.study = None
        self.best_params = None
        self.best_score = None
        
    def _evaluate_model(self, model, X, y, sample_weights) -> dict:
        auc_scores = []
        accuracy_scores = []
        
        # Determine if binary or multiclass
        n_classes = len(np.unique(y))
        is_binary = n_classes == 2
        
        for train_idx, test_idx in self.cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            w_train = sample_weights.iloc[train_idx]
            
            model.fit(
                X_train, y_train,
                sample_weight=w_train,
                # LightGBM specific: early stopping via callbacks is preferred in new versions, 
                # but for sklearn API standard fit is okay. 
                # We won't use early_stopping_rounds here to keep it simple with sklearn API wrapper
            )
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            
            if is_binary:
                # Handle case where only 1 class is present in test set
                if len(np.unique(y_test)) < 2:
                    auc = 0.5 # Neutral score
                else:
                    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                 auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            
            auc_scores.append(auc)
            accuracy_scores.append(acc)
            
        return {
            'auc_mean': np.mean(auc_scores),
            'auc_std': np.std(auc_scores),
            'accuracy_mean': np.mean(accuracy_scores)
        }

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective for LightGBM.
        """
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        metrics = self._evaluate_model(model, self.X, self.y, self.sample_weights)
        
        trial.set_user_attr('auc_std', metrics['auc_std'])
        return metrics['auc_mean']

    def optimize(self, n_trials=50):
        print(f"Starting LightGBM Optimization ({n_trials} trials)...")
        sampler = TPESampler(seed=42)
        self.study = optuna.create_study(direction='maximize', sampler=sampler, study_name='lgbm_opt')
        self.study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        print(f"Optimization finished. Best AUC: {self.best_score:.4f}")
        return self.best_params

    def train_best_model(self):
        print("Training final model with best parameters...")
        params = self.best_params.copy()
        params['objective'] = 'binary'
        params['random_state'] = 42
        params['n_jobs'] = -1
        # Add verbosity -1 to avoid warnings
        params['verbosity'] = -1
        
        model = lgb.LGBMClassifier(**params)
        model.fit(self.X, self.y, sample_weight=self.sample_weights)
        return model

def main():
    print("=" * 80)
    print("LightGBM Training & Optimization (Feature Engineering 2.0)")
    print("=" * 80)
    
    # 1. Load Data
    data_path = 'features_v2_labeled.csv'
    print(f"Loading {data_path}...")
    
    # Read with explicit index_col handling
    try:
        # Check first row to see if it looks like header
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Error reading csv: {e}")
        return

    # Check for t1
    if 't1' not in df.columns:
        print("   't1' column missing. Joining with labeled_events.csv...")
        try:
            events = pd.read_csv("labeled_events.csv", index_col=0, parse_dates=True)
            # Need to align indices
            df = df.join(events[['t1']], rsuffix='_events')
            if 't1' not in df.columns and 't1_events' in df.columns:
                df['t1'] = df['t1_events']
        except Exception as e:
            print(f"   Could not load t1 from events: {e}")
            return

    df['t1'] = pd.to_datetime(df['t1'])
    
    # Drop NaNs
    df = df.dropna()
    print(f"Data shape after cleaning: {df.shape}")
    
    # 2. Select Features
    feature_file = 'selected_features_v2.csv'
    if os.path.exists(feature_file):
        print(f"Using features from {feature_file}")
        sf = pd.read_csv(feature_file)
        # Assuming column 'feature' exists
        selected_cols = sf['feature'].tolist()
        # Intersect with df columns
        feature_cols = [c for c in selected_cols if c in df.columns]
        print(f"Selected {len(feature_cols)} features.")
    else:
        print("Warning: selected_features_v2.csv not found. Using all numeric columns.")
        feature_cols = df.select_dtypes(include=np.number).columns.tolist()
        exclude = ['label', 'ret', 'sample_weight', 'avg_uniqueness', 'bin']
        feature_cols = [c for c in feature_cols if c not in exclude]

    X = df[feature_cols]
    y = df['label']
    t1 = df['t1']
    
    # Sample Weights
    if 'sample_weight' in df.columns:
        sample_weights = df['sample_weight']
    else:
        sample_weights = None

    # 3. Optimize
    optimizer = LightGBMOptimizer(X, y, t1, sample_weights, n_splits=5)
    best_params = optimizer.optimize(n_trials=50)
    
    # 4. Save Params
    pd.DataFrame([best_params]).to_csv('best_hyperparameters_lgbm.csv', index=False)
    
    # 5. Train Final Model
    model = optimizer.train_best_model()
    
    # 6. Feature Importance
    fi = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fi.to_csv('feature_importance_lgbm_v2.csv', index=False)
    print("Saved feature importance to feature_importance_lgbm_v2.csv")
    
    # 7. Visualization
    os.makedirs('visual_analysis', exist_ok=True)
    plt.figure(figsize=(10, 10))
    import seaborn as sns
    sns.barplot(x='Importance', y='Feature', data=fi.head(20))
    plt.title('LightGBM Feature Importance (Top 20)')
    plt.tight_layout()
    plt.savefig('visual_analysis/feature_importance_lgbm.png')
    
    # 8. Report
    baseline_auc = 0.5122
    best_auc = optimizer.best_score
    print("-" * 50)
    print(f"Baseline (RF) AUC: {baseline_auc:.4f}")
    print(f"LightGBM V2 AUC:   {best_auc:.4f}")
    print(f"Improvement:       {(best_auc - baseline_auc)/baseline_auc*100:+.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    main()
