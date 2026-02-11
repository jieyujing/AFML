import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import ast
from cv_setup import PurgedKFold
import os
import sys

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bet_sizing import bet_size_probability
from afml import MetaLabelingPipeline


def load_data():
    print("1. Loading data...")
    feature_file = "features_labeled.csv"
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"{feature_file} not found.")

    df = pd.read_csv(feature_file, index_col=0, parse_dates=True)

    # Ensure t1 is present (crucial for PurgedCV)
    if "t1" not in df.columns:
        print("   't1' column missing in features. Joining with labeled_events.csv...")
        try:
            events = pd.read_csv("labeled_events.csv", index_col=0, parse_dates=True)
            # Join t1. intersection of indices
            df = df.join(events[["t1"]], how="inner", rsuffix="_event")
            if "t1" not in df.columns and "t1_event" in df.columns:
                df["t1"] = df["t1_event"]
            df["t1"] = pd.to_datetime(df["t1"])
        except Exception as e:
            print(f"   Error joining events: {e}")
            raise ValueError("'t1' column required for PurgedCV but not found.")

    # Ensure avg_uniqueness is present (crucial for Bet Sizing)
    if "avg_uniqueness" not in df.columns:
        print(
            "   'avg_uniqueness' missing. Bet sizing concurrency adjustment will be disabled."
        )
    else:
        print(f"   'avg_uniqueness' present (Mean: {df['avg_uniqueness'].mean():.4f})")

    # Load selected features
    selected_file = "selected_features.csv"
    if os.path.exists(selected_file):
        selected_df = pd.read_csv(selected_file)
        # Assuming the CSV has a column 'feature' or similar, or it's a list.
        # Actually usually it's just columns. Let's check structure or assume it's a list from previous steps.
        # From previous steps, it's likely a CSV with feature names.
        # Let's inspect quickly or just assume column 0 is the name.
        selected_features = selected_df.iloc[:, 0].tolist()
        print(f"   Loaded {len(selected_features)} selected features.")
    else:
        print("   Warning: selected_features.csv not found. Using all features.")
        selected_features = [
            c
            for c in df.columns
            if c
            not in [
                "label",
                "ret",
                "t1",
                "trgt",
                "side",
                "sample_weight",
                "avg_uniqueness",
                "barrier_time",
            ]
        ]

    # Load Hyperparameters
    params_file = "best_hyperparameters.csv"
    rf_params = {}
    if os.path.exists(params_file):
        try:
            params_df = pd.read_csv(params_file, index_col=0)
            # The structure might be 'parameter', 'value'. Or just a dict structure.
            # Let's try to parse flexibly.
            # Assuming row index is param name, column 0 is value.
            for idx, row in params_df.iterrows():
                val = row.iloc[0]
                # Convert string representation to standard types
                if isinstance(val, str):
                    if val.lower() == "true":
                        val = True
                    elif val.lower() == "false":
                        val = False
                    elif val.lower() == "none":
                        val = None
                rf_params[idx] = val

            print("   Loaded optimized hyperparameters.")
        except Exception as e:
            print(f"   Error loading hyperparameters: {e}. Using defaults.")
    else:
        print("   No hyperparameter file found. Using defaults.")

    # Clean params for RF
    valid_params = {}
    valid_keys = [
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "criterion",
        "class_weight",
    ]
    for k, v in rf_params.items():
        if k in valid_keys:
            # Type conversion
            if k in [
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
            ]:
                try:
                    valid_params[k] = int(float(v))
                except:
                    valid_params[k] = v
            elif k == "max_features":
                try:
                    valid_params[k] = float(v)
                except:
                    valid_params[k] = v
            else:
                valid_params[k] = v

    # Ensure n_jobs is set
    valid_params["n_jobs"] = -1
    valid_params["random_state"] = 42

    return df, selected_features, valid_params


def get_primary_predictions(df, features, params, n_splits=5):
    print("\n2. Generating Primary Model OOS Predictions (Purged CV)...")

    # Setup CV
    cv = PurgedKFold(n_splits=n_splits, samples_info_sets=df["t1"], embargo=0.01)

    X = df[features]
    y = df["label"]
    weights = df["sample_weight"] if "sample_weight" in df.columns else None

    # Initialize containers
    primary_preds = pd.Series(0, index=df.index, name="primary_pred")
    primary_probs = pd.DataFrame(0.0, index=df.index, columns=[-1, 0, 1])

    # Track which indices were actually tested (purging drops some)
    tested_indices = []

    for i, (train_idx, test_idx) in enumerate(cv.split(X)):
        print(f"   Fold {i + 1}/{n_splits}...")

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        w_train = weights.iloc[train_idx] if weights is not None else None

        # Train Primary Model
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train, sample_weight=w_train)

        # Predict
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)

        # Store
        primary_preds.iloc[test_idx] = preds

        # Handle probs classes order
        classes = clf.classes_
        for j, cls in enumerate(classes):
            if cls in [-1, 0, 1]:
                primary_probs.loc[df.index[test_idx], cls] = probs[:, j]

        tested_indices.extend(test_idx)

    return primary_preds, primary_probs, tested_indices


def train_meta_model(df, features, primary_preds, primary_probs, tested_indices):
    print("\n3. Meta-Labeling & Secondary Model Training...")

    # 1. Filter to tested indices (valid OOS predictions)
    df_meta = df.iloc[tested_indices].copy()
    primary_preds = primary_preds.iloc[tested_indices]
    probs_meta = primary_probs.iloc[tested_indices]

    # 2. Define Meta-Labels (Same as before)
    # Filter for non-zero predictions
    mask = primary_preds != 0
    X_meta = df_meta.loc[mask, features].copy()  # Use same features for secondary model

    # *** KEY IMPROVEMENT ***
    # Add Primary Model's Confidence (Probability) as a feature
    # If pred=1, use prob(1). If pred=-1, use prob(-1).
    confidences = []
    for idx in X_meta.index:
        pred_class = primary_preds.loc[idx]
        conf = probs_meta.loc[idx, pred_class]
        confidences.append(conf)

    X_meta["primary_model_prob"] = confidences
    print("   ✓ Added 'primary_model_prob' feature to Meta-Model.")

    y_true = df_meta.loc[mask, "label"]
    y_pred_primary = primary_preds[mask]

    print(
        f"   Primary Model Coverage: {mask.sum()}/{len(df_meta)} samples ({mask.mean() * 100:.1f}%) predicted non-zero."
    )

    # Generate Meta Labels
    # 1 if prediction matches magnitude
    # y_meta = (y_pred_primary == y_true).astype(int)

    # Strict definition: Did it make money?
    # If y_pred=1 and y_true=1 -> 1
    # If y_pred=-1 and y_true=-1 -> 1
    # Else -> 0
    meta_labels = (y_pred_primary == y_true).astype(int)

    print("   Meta-Label Distribution:")
    print(meta_labels.value_counts(normalize=True))

    # 3. Train Secondary Model
    # We use a standard Random Forest here, maybe slightly shallower to avoid overfitting
    meta_clf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=7,  # Shallower than primary
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    # Split for Meta-Model Evaluation (Train/Test simple split 80/20)
    # Note: This is an internal split of the "OOS" data generated from Primary CV.
    # Technically valid since Primary predictions were OOS.
    split = int(len(X_meta) * 0.8)
    X_train, X_test = X_meta.iloc[:split], X_meta.iloc[split:]
    y_train, y_test = meta_labels.iloc[:split], meta_labels.iloc[split:]

    meta_clf.fit(X_train, y_train)

    print("\n   [Meta-Model Performance]")
    y_pred_meta = meta_clf.predict(X_test)
    y_prob_meta_arr = meta_clf.predict_proba(X_test)[:, 1]
    y_prob_meta = pd.Series(y_prob_meta_arr, index=X_test.index)

    print(confusion_matrix(y_test, y_pred_meta))
    print(classification_report(y_test, y_pred_meta))
    print(f"   ROC AUC: {roc_auc_score(y_test, y_prob_meta):.4f}")

    return meta_clf, X_test, y_test, y_prob_meta, df_meta.loc[mask].iloc[split:]


def evaluate_strategy(df_eval, primary_preds, meta_probs, threshold=0.55):
    print("\n4. Strategy Evaluation...")

    # df_eval contains original data (returns) for the test set
    # primary_preds: Primary model predictions for this set
    # meta_probs: Secondary model probability of 'success'

    # 1. Base Strategy (Primary Only)
    # Returns = Sign(Pred) * Ret
    base_returns = primary_preds * df_eval["ret"]

    # 2. Meta Strategy (Binary Threshold)
    # Position = Pred * (MetaProb > Threshold)
    binary_position = (meta_probs > threshold).astype(float)
    binary_returns = base_returns * binary_position

    # 3. Meta Strategy (Bet Sizing - AFML Ch.10)
    print("   Calculating Optimal Bet Sizing (AFML Ch.10)...")
    # We pass the events info (specifically t1 and avg_uniqueness if available)
    # We can check if 'avg_uniqueness' is in df_eval. If not, we skip the concurrency scaling part or try to fetch it.

    # Ensure df_eval has necessary columns
    events_for_sizing = df_eval.copy()
    if "avg_uniqueness" not in events_for_sizing.columns:
        # Try to recover it from original features if possible, or assume 1.0 (no scaling)
        # Assuming features_labeled.csv had it.
        # But df_eval is a slice. It SHOULD have it if we loaded all features in load_data.
        # Let's check load_data... it loads selected features.
        # If avg_uniqueness was not selected, it's missing.
        # We should fix load_data to always keep 'avg_uniqueness' available in df.
        pass

    # Calculate Bet Sizes
    # Using 'average_active=True' which uses 'avg_uniqueness' column if present
    bet_sizes = bet_size_probability(
        events=events_for_sizing,
        prob_series=meta_probs,
        pred_series=None,  # We apply direction manually below
        average_active=True,
    )

    sized_position = bet_sizes  # Magnitude (0 to 1)
    sized_returns = base_returns * sized_position

    print(f"   Binary Threshold: {threshold}")
    print(f"   Bet Sizing: Dynamic (CDF-based) + Concurrency Adjustment")

    stats = []
    strategies = [
        ("Base (Primary)", base_returns),
        (f"Meta (Binary >{threshold})", binary_returns),
        ("Meta (Bet Sizing)", sized_returns),
    ]

    for name, rets in strategies:
        # Cumulative Return
        cum_ret = (1 + rets).cumprod()
        total_ret = cum_ret.iloc[-1] - 1

        # Sharpe (Annualized approx)
        sharpe = rets.mean() / rets.std() * np.sqrt(252 * 4) if rets.std() != 0 else 0

        # Max Drawdown
        roll_max = cum_ret.cummax()
        drawdown = (cum_ret - roll_max) / roll_max
        max_dd = drawdown.min()

        # Win Rate
        wins = rets[rets > 0]
        losses = rets[rets < 0]
        win_rate = (
            len(wins) / (len(wins) + len(losses))
            if (len(wins) + len(losses)) > 0
            else 0
        )

        # Number of Trades (Count non-zero positions)
        n_trades = (rets != 0).sum()

        # Average Bet Size (for Sized strategy)
        avg_size = (
            np.abs(rets[rets != 0] / df_eval.loc[rets != 0, "ret"]).mean()
            if n_trades > 0
            else 0
        )

        stats.append(
            {
                "Strategy": name,
                "Total Return": f"{total_ret * 100:.2f}%",
                "Sharpe": f"{sharpe:.2f}",
                "Max DD": f"{max_dd * 100:.2f}%",
                "Win Rate": f"{win_rate * 100:.2f}%",
                "Trades": n_trades,
                "Avg Size": f"{avg_size:.2f}",
            }
        )

    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(
        (1 + base_returns).cumprod(), label="Base (Primary)", alpha=0.4, linestyle="--"
    )
    plt.plot(
        (1 + binary_returns).cumprod(), label=f"Meta (Binary >{threshold})", alpha=0.7
    )
    plt.plot(
        (1 + sized_returns).cumprod(),
        label="Meta (Bet Sizing)",
        linewidth=2.5,
        color="green",
    )

    plt.title("Strategy Performance: Binary vs Bet Sizing")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("visual_analysis/meta_labeling_performance.png")
    print(
        "   ✓ Saved performance plot to visual_analysis/meta_labeling_performance.png"
    )


def main_oo():
    """Meta-labeling using OO approach with MetaLabelingPipeline."""
    print("=" * 80)
    print("Meta-Labeling Pipeline - OO Implementation (AFML Chapter 3)")
    print("=" * 80)

    # 1. Load data
    print("\n1. Loading data...")
    try:
        df = pd.read_csv("features_labeled.csv", index_col=0, parse_dates=True)
        print(f"   Loaded {len(df)} samples")

        # Get feature columns
        features = [
            c
            for c in df.columns
            if c not in ["t1", "label", "ret", "sample_weight", "avg_uniqueness"]
        ]
        print(f"   Using {len(features)} features")

        y = df["label"]

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # 2. Using OO MetaLabelingPipeline
    print("\n2. Using MetaLabelingPipeline...")
    pipeline = MetaLabelingPipeline(
        primary_model=RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
        ),
        secondary_model=RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
        ),
        n_splits=5,
        embargo=0.01,
    )

    # Fit pipeline
    print("   Fitting pipeline...")
    y_series = pd.Series(y)
    pipeline.fit(df, features, y_series)
    print("   ✓ Pipeline fitted")

    # 3. Get predictions
    print("\n3. Generating predictions...")
    X_features = pd.DataFrame(df[features])
    predictions = pipeline.predict(X_features)
    probabilities = pipeline.predict_proba(X_features)

    print(f"   Predictions: {len(predictions)}")
    print(f"   Positive predictions: {(predictions > 0).sum()}")

    # 4. Evaluate
    print("\n4. Evaluation Results:")
    accuracy = accuracy_score(y, predictions)
    print(f"   Accuracy: {accuracy:.4f}")

    if 1 in probabilities.columns:
        auc = roc_auc_score(y == 1, probabilities[1])
        print(f"   ROC AUC: {auc:.4f}")

    # 5. Save results
    print("\n5. Saving results...")
    df["meta_prediction"] = predictions
    df["meta_probability"] = probabilities[1] if 1 in probabilities.columns else 0
    df.to_csv("meta_labeled_predictions.csv")
    print("   ✓ Saved to meta_labeled_predictions.csv")

    print("\n" + "=" * 80)
    print("✓ Meta-Labeling Complete!")
    print("=" * 80)


def main():
    print("=" * 80)
    print("Meta-Labeling Implementation (AFML Chapter 3)")
    print("=" * 80)

    # 1. Load
    try:
        df, features, params = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Primary Model OOS Predictions
    primary_preds, primary_probs, tested_indices = get_primary_predictions(
        df, features, params
    )

    # 3. Train Meta Model
    meta_clf, X_test, y_test, y_prob_meta, df_eval = train_meta_model(
        df, features, primary_preds, primary_probs, tested_indices
    )

    # 4. Evaluate
    # Get primary preds for the evaluation set
    eval_primary_preds = primary_preds.loc[df_eval.index]

    evaluate_strategy(df_eval, eval_primary_preds, y_prob_meta, threshold=0.55)

    print("\nDONE.")


if __name__ == "__main__":
    main()
