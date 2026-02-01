import pandas as pd
import numpy as np
import os
import argparse
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from cv_setup import PurgedKFold  # Assumes src/ is in path or we run from root

def train_primary_model(
    input_file: str,
    output_file: str,
    n_estimators: int = 100,
    max_depth: int = 3
) -> pd.Series:
    """
    Train a simple directional (Primary) model using Purged K-Fold Cross Validation
    to generate Out-Of-Sample (OOS) predictions for the entire dataset.
    """
    
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return None

    # Check if we have labels and t1
    if 'label' not in df.columns:
        print("Error: 'label' column missing.")
        return None
    
    if 't1' not in df.columns:
        if 'barrier_time' in df.columns:
            df['t1'] = df['barrier_time']
        else:
            print("Error: 't1' column missing (required for Purged CV).")
            return None
        
    df['t1'] = pd.to_datetime(df['t1'])

    # Generate Simple Features for Primary Model
    print("Generating simple primary features...")
    
    # 1. Log Returns
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # 2. Momentum / Lags
    for lag in [1, 2, 3, 5, 10]:
        df[f'ret_lag_{lag}'] = df['log_ret'].shift(lag)
        
    # 3. Volatility
    df['volatility'] = df['log_ret'].rolling(window=20).std()
    
    # 4. Simple Moving Average Crossover
    df['sma_short'] = df['close'].rolling(window=10).mean()
    df['sma_long'] = df['close'].rolling(window=50).mean()
    df['trend'] = (df['sma_short'] > df['sma_long']).astype(int)
    
    # Drop NaNs created by lagging/rolling
    df.dropna(inplace=True)
    
    # Select Features
    feature_cols = [c for c in df.columns if 'lag' in c or c in ['volatility', 'trend']]
    X = df[feature_cols]
    y = df['label'] # {-1, 1}
    t1 = df['t1'] # For purging
    
    print(f"Features: {feature_cols}")
    print(f"Samples: {len(X)}")
    
    # Setup Purged CV
    n_splits = 5
    cv = PurgedKFold(n_splits=n_splits, samples_info_sets=t1, embargo=0.01)
    
    # Store OOS predictions
    # Initialize with 0 or NaN
    oos_preds = pd.Series(0, index=X.index, name='side')
    
    print(f"\nStarting Purged K-Folds (k={n_splits})...")
    
    for i, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        # Train
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Predict on Test
        y_pred = model.predict(X_test)
        
        # Store
        oos_preds.iloc[test_idx] = y_pred
        
        # Evaluate this fold
        acc = accuracy_score(y_test, y_pred)
        print(f"  Fold {i+1}: Accuracy = {acc:.4f}")
        
    # Evaluate Total OOS Performance
    print("\nOverall OOS Evaluation:")
    print(classification_report(y, oos_preds))
    print(f"Overall Accuracy: {accuracy_score(y, oos_preds):.4f}")
    
    # Save Side
    # Note: oos_preds has index aligned with df (which is dropna'd). 
    # The original file might have more rows (warmup period).
    # We should save what we have.
    
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    oos_preds.to_csv(output_file)
    print(f"\nSaved predicted side to {output_file}")
    
    return oos_preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input labeled bars")
    parser.add_argument("--output", type=str, required=True, help="Path to output side file")
    args = parser.parse_args()
    
    # Add src to path to allow importing cv_setup
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    
    train_primary_model(args.input, args.output)
