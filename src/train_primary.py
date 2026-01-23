import pandas as pd
import numpy as np
import os
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from typing import Optional

def train_primary_model(
    input_file: str,
    output_file: str,
    n_estimators: int = 100,
    max_depth: int = 3
) -> pd.DataFrame:
    """
    Train a simple directional (Primary) model.
    In AFML, the "Primary Model" often has high recall but low precision.
    Here we use a Random Forest on basic features (Ret, Vol) or Momemtum.
    
    Actually, to match AFML perfectly, the 'side' can be just a momentum indicator.
    But Step 5 explicitly says "Train a classifier".
    
    So we will:
    1. Load labeled_events_primary.csv (which contains t1, trgt, side, ret, label).
    2. But we need FEATURES.
       Since we might not have run full feature engineering, we will generate 
       simple features on the fly for the Primary Model:
       - Log Returns (lags 1, 2, 3)
       - Volatility
       - RSI / Momentum
       Alternatively, if 'dollar_bars_labeled_primary.csv' exists, it has OHLCV.
    """
    
    print(f"Loading data from {input_file}...")
    try:
        # Load the labeled bars (which include OHLCV + Label)
        df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return None

    # Check if we have labels
    if 'label' not in df.columns:
        print("Error: 'label' column missing.")
        return None

    # Generate Simple Features for Primary Model
    # (We don't need heavy FE 2.0 here, just enough to get direction)
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
    
    # Drop NaNs
    df.dropna(inplace=True)
    
    # Select Features
    feature_cols = [c for c in df.columns if 'lag' in c or c in ['volatility', 'trend']]
    X = df[feature_cols]
    y = df['label'] # {-1, 1} (assuming 0s filtered or rare)
    
    # Train-Test Split (Sequential)
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    print(f"Training Primary Model (RF, n={n_estimators}, depth={max_depth})...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("\nPrimary Model Evaluation (Test Set):")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Generate 'Side' signal for the FULL dataset (or at least where we have features)
    # The 'Side' is the prediction of the Primary Model.
    # We use this 'Side' to decide whether to take a Macro Bet (Meta Labeling).
    
    # We want predictions for the WHOLE dataframe (X).
    # Note: This is in-sample for the first 80%, but that's standard for 
    # generating the "side" which effectively becomes a feature/filter for the secondary model.
    # (Or better yet, use a rolling window, but for simplicity here we use the full model inference).
    
    full_preds = model.predict(X)
    
    # Create Side Series
    # Map predictions to Side: 1 (Long), -1 (Short)
    # The labels from labeling.py 1:1 are already {-1, 1}.
    side_series = pd.Series(full_preds, index=X.index, name='side')
    
    # Save Side
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    side_series.to_csv(output_file)
    print(f"\nSaved predicted side to {output_file}")
    
    return side_series

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input labeled bars")
    parser.add_argument("--output", type=str, required=True, help="Path to output side file")
    args = parser.parse_args()
    
    train_primary_model(args.input, args.output)
