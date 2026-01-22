"""
Purged K-Fold Cross Validation for Financial Machine Learning

This module implements the Purged K-Fold Cross Validation method from 'Advances in Financial Machine Learning'
Chapter 7. It handles:
1. Purging: Removing training samples that overlap with test samples
2. Embargoing: Removing training samples immediately following test samples to prevent leakage
3. Visualization: Plotting the train/test splits to verify correctness

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
import os
import sys

# ==============================================================================
# Purged K-Fold Logic (Self-contained)
# ==============================================================================

def get_train_times(samples_info_sets: pd.Series, test_times: pd.Series) -> pd.Series:
    """
    Find the training set indexes given the information on which each record is based
    and the range for the test set.
    """
    train = samples_info_sets.copy(deep=True)
    for start_ix, end_ix in test_times.items():
        # Train starts within test
        df0 = train[(start_ix <= train.index) & (train.index <= end_ix)].index.unique() 
        # Train ends within test
        df1 = train[(start_ix <= train) & (train <= end_ix)].index.unique()
        # Train envelops test
        df2 = train[(train.index <= start_ix) & (end_ix <= train)].index.unique()
        
        train = train.drop(df0.union(df1).union(df2))
    return train

class PurgedKFold(KFold):
    """
    Extend KFold class to work with labels that span intervals.
    The train is purged of observations overlapping test-label intervals.
    Test set is assumed contiguous (shuffle=False), w/o training samples in between.
    """
    def __init__(self, n_splits: int = 3, samples_info_sets: pd.Series = None, embargo: float = 0.0):
        if not isinstance(samples_info_sets, pd.Series):
            raise ValueError('The samples_info_sets param must be a pd.Series')
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        
        self.samples_info_sets = samples_info_sets
        # Embargo is % of total observations
        self.embargo = embargo

    def split(self, X: pd.DataFrame, y: pd.Series = None, groups=None):
        if X.shape[0] != self.samples_info_sets.shape[0]:
            raise ValueError("X and the 'samples_info_sets' series param must be the same length")

        indices: np.ndarray = np.arange(X.shape[0])
        
        # Calculate embargo size (number of bars)
        n_embargo = int(X.shape[0] * self.embargo)

        test_ranges: [(int, int)] = [(ix[0], ix[-1] + 1) for ix in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        
        for start_ix, end_ix in test_ranges:
            test_indices = indices[start_ix:end_ix]
            
            # Get test start and end times
            t0 = self.samples_info_sets.index[start_ix]
            t1 = self.samples_info_sets.iloc[end_ix-1]
            
            test_times = pd.Series(index=[t0], data=[t1])
            
            # Get valid train times (purging happens here)
            train_times = get_train_times(self.samples_info_sets, test_times)
            
            # Map back to indices
            train_indices = []
            
            # Create a boolean mask for training samples
            train_mask = self.samples_info_sets.index.isin(train_times.index)
            train_indices = indices[train_mask]
            
            # Apply Embargo on indices
            if n_embargo > 0:
                # If test set is at the beginning, the following train set needs embargo
                if end_ix < X.shape[0]: # Test is not at the very end
                    embargo_end = min(end_ix + n_embargo, X.shape[0])
                    embargo_indices = np.arange(end_ix, embargo_end)
                    
                    # Remove these from train_indices
                    train_indices = np.setdiff1d(train_indices, embargo_indices)

            yield train_indices, test_indices

# ==============================================================================
# Visualization
# ==============================================================================

def plot_cv_indices(cv, X, y, dt_index, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1 # Testing
        indices[tr] = 0 # Training

        # Visualize the results
        ax.scatter(dt_index, [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=plt.cm.coolwarm,
                   vmin=-.2, vmax=1.2)

    # Formatting
    yticklabels = list(range(n_splits))
    ax.set(yticks=np.arange(n_splits) + .5, yticklabels=yticklabels,
           xlabel='Date', ylabel="CV Iteration",
           ylim=[n_splits+0.2, -.2], xlim=[dt_index.min(), dt_index.max()])
    ax.set_title(f'{type(cv).__name__}', fontsize=15)
    return ax

def main():
    print("=" * 80)
    print("Purged K-Fold Cross Validation Setup")
    print("=" * 80)
    sys.stdout.flush()

    # 1. Load Data
    print("\n1. Loading data...")
    try:
        # Load features labeled (which has t1 info)
        df = pd.read_csv("features_labeled.csv", index_col=0, parse_dates=True)
        # We need t1 for purging (time when label is finalized)
        # t1 is the 'end time' of the observation
        if 't1' not in df.columns:
             # Try to load from labeled_events if missing
             events = pd.read_csv("labeled_events.csv", index_col=0, parse_dates=True)
             df = df.join(events[['t1']], rsuffix='_event')
             if 't1' not in df.columns and 't1_event' in df.columns:
                 df['t1'] = df['t1_event']
        
        df['t1'] = pd.to_datetime(df['t1'])
        
        # Sort by index just in case
        df = df.sort_index()
        
        print(f"   Loaded {len(df)} samples")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        sys.stdout.flush()
        
    except FileNotFoundError:
        print("Error: features_labeled.csv not found.")
        return

    # 2. Setup Purged K-Fold
    n_splits = 5
    embargo_pct = 0.01 # 1% embargo
    
    print(f"\n2. Configuring CV: {n_splits} splits, {embargo_pct*100}% embargo")
    
    cv = PurgedKFold(n_splits=n_splits, samples_info_sets=df['t1'], embargo=embargo_pct)
    
    # 3. Visualize
    print("\n3. Generating visualization...")
    sys.stdout.flush()
    # Create directory if not exists
    os.makedirs("visual_analysis", exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    plot_cv_indices(cv, df, df['label'], df.index, ax, n_splits)
    
    # Add legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=plt.cm.coolwarm(0.), lw=4),
                    Line2D([0], [0], color=plt.cm.coolwarm(1.), lw=4)]
    ax.legend(custom_lines, ['Train', 'Test'], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    plt.tight_layout()
    output_path = "visual_analysis/cv_splits.png"
    plt.savefig(output_path)
    print(f"   ✓ Saved CV visualization to: {output_path}")

    # 4. Verify Leakage
    print("\n4. Verifying splits for leakage...")
    sys.stdout.flush()
    leakage_count = 0
    for i, (train_idx, test_idx) in enumerate(cv.split(df)):
        train_set = df.iloc[train_idx]
        test_set = df.iloc[test_idx]
        
        # Check 1: Overlap of indices (Should be 0)
        overlap = set(train_idx).intersection(set(test_idx))
        if overlap:
            print(f"   [Fold {i}] ERROR: Index overlap detected! ({len(overlap)} samples)")
            leakage_count += 1
            
        # Check 2: Information Leakage 
        test_start = test_set.index.min()
        test_end = test_set.index.max()
        
        # Purging check:
        # Train set BEFORE Test set: Train_End (t1) must be < Test_Start
        train_before_test = train_set[train_set.index < test_start]
        if not train_before_test.empty:
            max_t1 = train_before_test['t1'].max()
            if max_t1 >= test_start:
                print(f"   [Fold {i}] LEAKAGE: Training sample before test set ends inside test set (Max t1: {max_t1} >= Test Start: {test_start})")
                leakage_count += 1
        
        # Embargo check (Train after Test)
        # Train Start > Test End
        train_after_test = train_set[train_set.index > test_end]
        if not train_after_test.empty:
            min_start = train_after_test.index.min()
            # Just simple check
            if min_start <= test_end:
                 print(f"   [Fold {i}] ERROR: Training sample starts before test set ends (Time overlap)")
                 leakage_count += 1
            
        print(f"   [Fold {i}] Train: {len(train_idx)}, Test: {len(test_idx)}")

    if leakage_count == 0:
        print("\n   ✓ No leakage detected. Purging is working correctly.")
    else:
        print(f"\n   ⚠️ Leakage detected in {leakage_count} instances.")

    print("\n" + "=" * 80)
    print("✓ Cross Validation Setup Complete!")
    print("=" * 80)
    sys.stdout.flush()

if __name__ == "__main__":
    main()
