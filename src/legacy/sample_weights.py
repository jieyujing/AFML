"""
Sample Weights Calculation for Financial Machine Learning

This module implements sample weighting strategies from 'Advances in Financial Machine Learning'
Chapter 4. It calculates:
1. Concurrency: How many labels overlap at each time point
2. Average Uniqueness: How unique each label is (inverse of concurrency)
3. Sample Weights: Final weights based on uniqueness and returns

These weights are crucial for addressing the non-IID nature of financial data
where overlapping labels introduce redundancy.

NOTE: This file includes a self-contained implementation of the necessary logic
from mlfinlab to avoid heavy dependencies (like numba) that are problematic
in some environments.
"""

import pandas as pd
import numpy as np
import sys
import os
import time
import datetime as dt
import multiprocessing as mp
from pathlib import Path

from afml import SampleWeightCalculator

# ==============================================================================
# Multiprocessing Utilities (Adapted from mlfinlab/util/multiprocess.py)
# ==============================================================================


def lin_parts(num_atoms, num_threads):
    parts = np.linspace(0, num_atoms, min(num_threads, num_atoms) + 1)
    parts = np.ceil(parts).astype(int)
    return parts


def nested_parts(num_atoms, num_threads, upper_triangle=False):
    parts = [0]
    num_threads_ = min(num_threads, num_atoms)
    for _ in range(num_threads_):
        part = 1 + 4 * (
            parts[-1] ** 2 + parts[-1] + num_atoms * (num_atoms + 1.0) / num_threads_
        )
        part = (-1 + part**0.5) / 2.0
        parts.append(part)
    parts = np.round(parts).astype(int)
    if upper_triangle:
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)
    return parts


def expand_call(kargs):
    func = kargs["func"]
    del kargs["func"]
    out = func(**kargs)
    return out


def report_progress(job_num, num_jobs, time0, task):
    msg = [float(job_num) / num_jobs, (time.time() - time0) / 60.0]
    msg.append(msg[1] * (1 / msg[0] - 1))
    time_stamp = str(dt.datetime.fromtimestamp(time.time()))
    msg = (
        time_stamp
        + " "
        + str(round(msg[0] * 100, 2))
        + "% "
        + task
        + " done after "
        + str(round(msg[1], 2))
        + " minutes. Remaining "
        + str(round(msg[2], 2))
        + " minutes."
    )
    if job_num < num_jobs:
        sys.stderr.write(msg + "\r")
    else:
        sys.stderr.write(msg + "\n")


def process_jobs_(jobs):
    out = []
    for job in jobs:
        out_ = expand_call(job)
        out.append(out_)
    return out


def process_jobs(jobs, task=None, num_threads=24, verbose=True):
    if task is None:
        task = jobs[0]["func"].__name__
    pool = mp.Pool(processes=num_threads)
    outputs = pool.imap_unordered(expand_call, jobs)
    out = []
    time0 = time.time()
    for i, out_ in enumerate(outputs, 1):
        out.append(out_)
        if verbose:
            report_progress(i, len(jobs), time0, task)
    pool.close()
    pool.join()
    return out


def mp_pandas_obj(
    func, pd_obj, num_threads=24, mp_batches=1, lin_mols=True, verbose=True, **kargs
):
    if lin_mols:
        parts = lin_parts(len(pd_obj[1]), num_threads * mp_batches)
    else:
        parts = nested_parts(len(pd_obj[1]), num_threads * mp_batches)

    jobs = []
    for i in range(1, len(parts)):
        job = {pd_obj[0]: pd_obj[1][parts[i - 1] : parts[i]], "func": func}
        job.update(kargs)
        jobs.append(job)

    if num_threads == 1:
        out = process_jobs_(jobs)
    else:
        out = process_jobs(jobs, num_threads=num_threads, verbose=verbose)

    if isinstance(out[0], pd.DataFrame):
        df0 = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        df0 = pd.Series(dtype="float64")
    else:
        return out

    for i in out:
        df0 = pd.concat([df0, i])  # Changed from append to concat for newer pandas

    df0 = df0.sort_index()
    return df0


# ==============================================================================
# Concurrency & Uniqueness (Adapted from mlfinlab/sampling/concurrent.py)
# ==============================================================================


def num_concurrent_events(close_series_index, label_endtime, molecule):
    label_endtime = label_endtime.fillna(close_series_index[-1])
    label_endtime = label_endtime[label_endtime >= molecule[0]]
    label_endtime = label_endtime.loc[: label_endtime[molecule].max()]

    nearest_index = close_series_index.searchsorted(
        pd.DatetimeIndex([label_endtime.index[0], label_endtime.max()])
    )
    # Handle searchsorted result which might be scalar or array
    idx_start = (
        nearest_index[0]
        if isinstance(nearest_index, (list, np.ndarray))
        else nearest_index
    )
    idx_end = (
        nearest_index[1]
        if isinstance(nearest_index, (list, np.ndarray)) and len(nearest_index) > 1
        else -1
    )

    # If using searchsorted on DatetimeIndex, it returns positions.
    # Correct logic:
    # Get the range of close prices that cover the events in the molecule
    start_time = label_endtime.index[0]
    end_time = label_endtime.max()

    # Use slicing instead of searchsorted to be safer with pandas versions
    relevant_close = close_series_index[
        (close_series_index >= start_time) & (close_series_index <= end_time)
    ]

    count = pd.Series(0, index=relevant_close)
    for t_in, t_out in label_endtime.items():  # iteritems -> items in newer pandas
        count.loc[t_in:t_out] += 1
    return count.loc[molecule[0] : label_endtime[molecule].max()]


def _get_average_uniqueness(label_endtime, num_conc_events, molecule):
    wght = pd.Series(index=molecule, dtype="float64")
    for t_in, t_out in label_endtime.loc[wght.index].items():
        wght.loc[t_in] = (1.0 / num_conc_events.loc[t_in:t_out]).mean()
    return wght


def get_av_uniqueness_from_triple_barrier(
    triple_barrier_events, close_series, num_threads, verbose=True
):
    out = pd.DataFrame()
    num_conc_events = mp_pandas_obj(
        num_concurrent_events,
        ("molecule", triple_barrier_events.index),
        num_threads,
        close_series_index=close_series.index,
        label_endtime=triple_barrier_events["t1"],
        verbose=verbose,
    )
    num_conc_events = num_conc_events.loc[
        ~num_conc_events.index.duplicated(keep="last")
    ]
    num_conc_events = num_conc_events.reindex(close_series.index).fillna(0)
    out["tW"] = mp_pandas_obj(
        _get_average_uniqueness,
        ("molecule", triple_barrier_events.index),
        num_threads,
        label_endtime=triple_barrier_events["t1"],
        num_conc_events=num_conc_events,
        verbose=verbose,
    )
    return out


# ==============================================================================
# Attribution (Adapted from mlfinlab/sample_weights/attribution.py)
# ==============================================================================


def _apply_weight_by_return(label_endtime, num_conc_events, close_series, molecule):
    ret = np.log(close_series).diff()
    weights = pd.Series(index=molecule, dtype="float64")

    for t_in, t_out in label_endtime.loc[weights.index].items():
        weights.loc[t_in] = (
            ret.loc[t_in:t_out] / num_conc_events.loc[t_in:t_out]
        ).sum()
    return weights.abs()


def get_weights_by_return(
    triple_barrier_events, close_series, num_threads=5, verbose=True
):
    # num_concurrent_events logic is duplicated here in original code, but we can reuse if we had it.
    # But `get_weights_by_return` orchestrator re-runs it.

    num_conc_events = mp_pandas_obj(
        num_concurrent_events,
        ("molecule", triple_barrier_events.index),
        num_threads,
        close_series_index=close_series.index,
        label_endtime=triple_barrier_events["t1"],
        verbose=verbose,
    )
    num_conc_events = num_conc_events.loc[
        ~num_conc_events.index.duplicated(keep="last")
    ]
    num_conc_events = num_conc_events.reindex(close_series.index).fillna(0)

    weights = mp_pandas_obj(
        _apply_weight_by_return,
        ("molecule", triple_barrier_events.index),
        num_threads,
        label_endtime=triple_barrier_events["t1"],
        num_conc_events=num_conc_events,
        close_series=close_series,
        verbose=verbose,
    )
    weights *= weights.shape[0] / weights.sum()
    return weights


# ==============================================================================
# Main Execution
# ==============================================================================


def main():
    """Calculate sample weights using OO approach."""
    print("=" * 80)
    print("Sample Weights Calculation - AFML Chapter 4")
    print("=" * 80)

    # 1. Load Data
    print("\n1. Loading data...")
    try:
        # Load full price series (needed for concurrency calculation across full history)
        close_df = pd.read_csv("dynamic_dollar_bars.csv", index_col=0, parse_dates=True)
        close_series = close_df["close"]

        # Load labeled events
        events = pd.read_csv("labeled_events.csv", index_col=0, parse_dates=True)

        # Ensure t1 is datetime
        events["t1"] = pd.to_datetime(events["t1"])

        print(f"   Loaded {len(close_series)} price bars")
        print(f"   Loaded {len(events)} labeled events")

    except FileNotFoundError as e:
        print(f"Error: Required file not found. {e}")
        return

    # 2. Using OO SampleWeightCalculator
    print("\n2. Using SampleWeightCalculator...")
    calculator = SampleWeightCalculator(decay=0.9)

    # Fit with events (needs 't1' column)
    calculator.fit(events)

    # Get uniqueness
    uniqueness = calculator.uniqueness_
    print("   Analysis of Uniqueness:")
    print(f"   Mean Uniqueness: {uniqueness.mean():.4f}")
    print(f"   Min Uniqueness:  {uniqueness.min():.4f}")
    print(f"   Max Uniqueness:  {uniqueness.max():.4f}")

    # 3. Transform to get weights
    print("\n3. Calculating Sample Weights...")
    result_df = calculator.transform(events)

    weights = result_df["sample_weight"]
    print("   Analysis of Weights:")
    print(f"   Mean Weight: {weights.mean():.4f}")
    print(f"   Min Weight:  {weights.min():.4f}")
    print(f"   Max Weight:  {weights.max():.4f}")
    print(f"   Count: {len(weights)}")

    # 4. Save Results
    print("\n4. Saving results...")

    output_file = "sample_weights.csv"
    result_df.to_csv(output_file)
    print(f"   ✓ Saved weights to: {output_file}")

    # 5. Update features_labeled.csv if it exists
    features_file = "features_labeled.csv"
    if os.path.exists(features_file):
        print(f"\n5. Updating {features_file} with weights...")
        features_df = pd.read_csv(features_file, index_col=0, parse_dates=True)

        # Join weights
        features_df["sample_weight"] = result_df["sample_weight"]
        features_df["avg_uniqueness"] = result_df["avg_uniqueness"]

        # Handle NaNs if any
        null_weights = features_df["sample_weight"].isnull().sum()
        if null_weights > 0:
            print(
                f"   Warning: {null_weights} rows in features file missing weights (filled with 0)"
            )
            features_df["sample_weight"] = features_df["sample_weight"].fillna(0)
            features_df["avg_uniqueness"] = features_df["avg_uniqueness"].fillna(0)

        features_df.to_csv(features_file)
        print(f"   ✓ Updated {features_file}")

    print("\n" + "=" * 80)
    print("✓ Sample Weights Calculation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
