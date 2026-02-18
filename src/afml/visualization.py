"""
Visualization Module.

This module provides tools for visualizing each step of the AFML pipeline,
including bar statistics, triple-barrier events, label distributions,
feature correlations, stationarity checks, cross-validation splits,
and strategy performance metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from scipy import stats

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_context("paper")


def _compute_jb_statistics(returns: np.ndarray) -> Dict[str, Any]:
    """
    Compute Jarque-Bera statistics for normality testing.

    Args:
        returns: Array of log returns.

    Returns:
        Dictionary containing JB statistics:
        - jb_stat: Jarque-Bera statistic
        - p_value: JB test p-value
        - skewness: Sample skewness
        - kurtosis: Sample kurtosis (Pearson, normal=3)
        - is_normal: True if p_value > 0.05
        - n_samples: Number of samples
    """
    # Remove NaN values
    returns = returns[np.isfinite(returns)]
    n = len(returns)

    if n < 4:
        return {
            "jb_stat": 0.0,
            "p_value": 1.0,
            "skewness": 0.0,
            "kurtosis": 3.0,
            "is_normal": True,
            "n_samples": n,
        }

    # Compute statistics
    jb_stat, p_value = stats.jarque_bera(returns)
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns, fisher=False)  # Pearson kurtosis (normal=3)

    return {
        "jb_stat": float(jb_stat),
        "p_value": float(p_value),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "is_normal": bool(p_value > 0.05),
        "n_samples": n,
    }


class AFMLVisualizer:
    """
    Visualizer for AFML Pipeline artifacts.
    """

    def __init__(self, output_dir: str = "visual_analysis"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_plot(self, filename: str):
        """Save current plot to file."""
        path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"  Saved plot: {path}")

    def plot_bar_stats(
        self,
        bars_df: pl.DataFrame,
        filename: str = "bar_stats.png",
        time_bars_df: Optional[pl.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Plot bar statistics (count, volume, dollar value) with JB statistics.

        Args:
            bars_df: Dollar bars DataFrame.
            filename: Output filename.
            time_bars_df: Optional time bars DataFrame for comparison.

        Returns:
            Dictionary containing JB statistics for dollar bars.
        """
        if bars_df.is_empty():
            return {"error": "Empty DataFrame"}

        # Increase figure height to accommodate JB statistics table and volume
        fig, axes = plt.subplots(5, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [1, 1, 2, 1, 1]})

        # Convert to pandas for easier plotting with seaborn/matplotlib
        pdf = bars_df.to_pandas()

        # 1. Daily Bar Counts
        daily_counts = pdf.set_index("datetime").resample("D").size()
        daily_counts = daily_counts[daily_counts > 0]

        axes[0].bar(daily_counts.index, daily_counts.values, alpha=0.7)
        axes[0].set_title("Number of Bars per Day")
        axes[0].set_ylabel("Count")

        # 2. Log Returns Distribution
        returns = np.log(pdf["close"] / pdf["close"].shift(1)).dropna()
        sns.histplot(returns, bins=50, kde=True, ax=axes[1])
        axes[1].set_title("Log Returns Distribution")
        axes[1].set_xlabel("Log Return")

        # 3. Candlestick Chart (Last N bars)
        ax = axes[2]
        N = 100  # Show last N bars for clarity

        # Get data slice
        ohlc = pdf.tail(N).reset_index(drop=True)

        if not ohlc.empty and all(
            c in ohlc.columns for c in ["open", "high", "low", "close"]
        ):
            # Up candles (Close >= Open)
            up = ohlc[ohlc["close"] >= ohlc["open"]]

            # Down candles (Close < Open)
            down = ohlc[ohlc["close"] < ohlc["open"]]

            # Plot Up candles
            if not up.empty:
                ax.bar(
                    up.index,
                    up["close"] - up["open"],
                    bottom=up["open"],
                    color="green",
                    width=0.6,
                )
                ax.vlines(up.index, up["low"], up["high"], color="green", linewidth=1)

            # Plot Down candles
            if not down.empty:
                ax.bar(
                    down.index,
                    down["close"] - down["open"],
                    bottom=down["open"],
                    color="red",
                    width=0.6,
                )
                ax.vlines(
                    down.index, down["low"], down["high"], color="red", linewidth=1
                )

            ax.set_title(f"Candlestick Chart (Last {len(ohlc)} Bars)")
            ax.set_ylabel("Price")
            # Remove x-labels for candlestick as volume is below
            ax.set_xticks([]) 
            ax.grid(True, alpha=0.3)

            # 4. Volume Chart (Last N bars)
            ax_vol = axes[3]
            # Color volume bars based on close >= open
            colors = ["green" if c >= o else "red" for c, o in zip(ohlc["close"], ohlc["open"])]
            ax_vol.bar(ohlc.index, ohlc["volume"], color=colors, alpha=0.6, width=0.6)
            ax_vol.set_title("Volume")
            ax_vol.set_ylabel("Volume")
            ax_vol.set_xlabel("Bar Index")
            ax_vol.set_xlim(ax.get_xlim()) # Sync x-axis limits
            ax_vol.grid(True, alpha=0.3)

        else:
            ax.text(0.5, 0.5, "Insufficient Data for K-line", ha="center")
            axes[3].text(0.5, 0.5, "Insufficient Data for Volume", ha="center")

        # 5. JB Statistics Panel
        ax_stats = axes[4]
        ax_stats.axis("off")

        # Compute JB statistics for dollar bars
        dollar_jb = _compute_jb_statistics(returns.values)

        # Compute JB statistics for time bars if provided
        time_jb = None
        if time_bars_df is not None and not time_bars_df.is_empty():
            time_pdf = time_bars_df.to_pandas()
            time_returns = np.log(
                time_pdf["close"] / time_pdf["close"].shift(1)
            ).dropna()
            time_jb = _compute_jb_statistics(time_returns.values)

        # Create comparison table
        if time_jb is not None:
            # Comparison mode
            table_data = [
                [
                    "JB Statistic",
                    f"{dollar_jb['jb_stat']:.2f}",
                    f"{time_jb['jb_stat']:.2f}",
                ],
                ["p-value", f"{dollar_jb['p_value']:.4f}", f"{time_jb['p_value']:.4f}"],
                [
                    "Skewness",
                    f"{dollar_jb['skewness']:.4f}",
                    f"{time_jb['skewness']:.4f}",
                ],
                [
                    "Kurtosis",
                    f"{dollar_jb['kurtosis']:.4f}",
                    f"{time_jb['kurtosis']:.4f}",
                ],
                [
                    "Normality",
                    "Pass" if dollar_jb["is_normal"] else "Fail",
                    "Pass" if time_jb["is_normal"] else "Fail",
                ],
                ["Samples", str(dollar_jb["n_samples"]), str(time_jb["n_samples"])],
            ]
            col_labels = ["Metric", "Dollar Bars", "Time Bars"]
        else:
            # Single mode (dollar bars only)
            table_data = [
                ["JB Statistic", f"{dollar_jb['jb_stat']:.2f}"],
                ["p-value", f"{dollar_jb['p_value']:.4f}"],
                ["Skewness", f"{dollar_jb['skewness']:.4f}"],
                ["Kurtosis", f"{dollar_jb['kurtosis']:.4f}"],
                ["Normality", "Pass" if dollar_jb["is_normal"] else "Fail"],
                ["Samples", str(dollar_jb["n_samples"])],
            ]
            col_labels = ["Metric", "Dollar Bars"]

        # Create table
        table = ax_stats.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
            colColours=["#4472C4", "#4472C4", "#4472C4"]
            if time_jb
            else ["#4472C4", "#4472C4"],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)

        # Style header cells
        for j in range(len(col_labels)):
            table[(0, j)].set_text_props(color="white", fontweight="bold")

        ax_stats.set_title(
            "Jarque-Bera Statistics (Normality Test)", pad=20, fontweight="bold"
        )

        # Add AFML compliance conclusion if time bars provided
        if time_jb is not None:
            if dollar_jb["jb_stat"] < time_jb["jb_stat"]:
                conclusion = f"[PASS] Dollar Bars reduce JB statistic: {dollar_jb['jb_stat']:.2f} < {time_jb['jb_stat']:.2f}"
                conclusion_color = "green"
            else:
                conclusion = "[FAIL] Dollar Bars do not improve normality"
                conclusion_color = "red"

            ax_stats.text(
                0.5,
                -0.15,
                conclusion,
                transform=ax_stats.transAxes,
                fontsize=12,
                fontweight="bold",
                color=conclusion_color,
                ha="center",
                va="top",
            )

            # Add reference for normal distribution
            ref_text = "Reference: Normal distribution has Skewness=0, Kurtosis=3, JB Statistic=0"
            ax_stats.text(
                0.5,
                -0.25,
                ref_text,
                transform=ax_stats.transAxes,
                fontsize=10,
                color="gray",
                ha="center",
                va="top",
            )

        self.save_plot(filename)

        return dollar_jb

    def plot_triple_barrier_sample(
        self,
        price_df: pl.DataFrame,
        events: pl.DataFrame,
        labels: pl.DataFrame,
        n_samples: int = 200,
        filename: str = "triple_barrier_sample.png",
    ):
        """
        Visualize Triple Barrier events on price series.

        Args:
            price_df: Dollar bars DataFrame with 'close'.
            events: Events DataFrame (unused here as labels contain t0).
            labels: DataFrame with 't0', 't1', 'label', 'tr'.
            n_samples: Number of bars to plot.
            filename: Output filename.
        """
        if price_df.is_empty() or labels.is_empty():
            return

        if "t0" not in labels.columns:
            # Fallback if t0 missing (e.g. older labeler version)
            print(
                "  [Viz] Warning: 't0' column missing in labels. Skipping triple barrier plot."
            )
            return

        # Pick a window with events
        # Sort labels by t0
        labels = labels.sort("t0")

        # Find a middle event
        mid_idx = len(labels) // 2
        center_t0 = labels["t0"][mid_idx]

        start_idx = max(0, center_t0 - n_samples // 2)
        end_idx = min(len(price_df), start_idx + n_samples)

        # Slice price data
        price_slice = price_df[start_idx:end_idx]
        price_pdf = price_slice.to_pandas()

        # Slice relevant labels
        # t0 in [start_idx, end_idx)
        labels_slice = labels.filter(
            (pl.col("t0") >= start_idx) & (pl.col("t0") < end_idx)
        )

        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot price
        # x-axis: range(start_idx, end_idx) or range(len(price_slice))
        # Let's use relative index for simplicity or datetime if available
        x_values = range(start_idx, end_idx)

        ax.plot(
            x_values, price_pdf["close"], label="Close Price", alpha=0.6, color="gray"
        )

        # Plot Barriers and Labels
        for row in labels_slice.iter_rows(named=True):
            t0 = row["t0"]
            t1 = row["t1"]
            label = row["label"]

            # Color code
            color = "green" if label == 1 else "red" if label == -1 else "blue"
            marker = "^" if label == 1 else "v" if label == -1 else "o"

            # Plot entry
            entry_price = price_df["close"][t0]
            ax.scatter(t0, entry_price, c="black", marker=".", s=50)  # Entry point

            # Plot exit/barrier touch
            if t1 < len(price_df):
                exit_price = price_df["close"][t1]
                # Only plot if within view (t1 can be outside but t0 is inside)
                if t1 < end_idx:
                    ax.scatter(
                        t1,
                        exit_price,
                        c=color,
                        marker=marker,
                        s=80,
                        label=f"Label {label}"
                        if f"Label {label}" not in ax.get_legend_handles_labels()[1]
                        else "",
                    )
                    # Draw line connecting
                    ax.plot(
                        [t0, t1],
                        [entry_price, exit_price],
                        c=color,
                        alpha=0.3,
                        linestyle="--",
                    )

        ax.set_title(f"Triple Barrier Labels (Sample Window: {start_idx}-{end_idx})")
        ax.set_xlabel("Bar Index")
        ax.set_ylabel("Price")
        ax.legend()

        self.save_plot(filename)

    def plot_label_distribution(
        self, labeled_df: pl.DataFrame, filename: str = "label_distribution.png"
    ):
        """
        Plot distribution of labels with detailed statistics.

        Args:
            labeled_df: DataFrame with 'label' and 'tr'/'ret'.
            filename: Output filename.
        """
        if labeled_df.is_empty() or "label" not in labeled_df.columns:
            return

        # Create layout: 2 plots + 1 stats panel
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.6])

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])

        pdf = labeled_df.to_pandas()

        # 1. Count Plot
        # Ensure labels are integers for consistent plotting
        try:
            pdf["label"] = pdf["label"].astype(int)
        except Exception:
            pass  # Fallback if conversion fails

        # Use simple colors: Red (-1), Blue (0), Green (1)
        # Add string keys for robustness
        palette = {-1: "red", 0: "blue", 1: "green", "-1": "red", "0": "blue", "1": "green"}
        sns.countplot(x="label", data=pdf, ax=ax1, palette=palette, hue="label", legend=False)
        ax1.set_title("Label Counts")

        # Add counts on top of bars
        for p in ax1.patches:
            height = int(p.get_height())
            ax1.annotate(
                f"{height}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
            )

        # 2. Return Distribution by Label
        return_col = "tr" if "tr" in pdf.columns else "ret"
        if return_col in pdf.columns:
            sns.boxplot(x="label", y=return_col, data=pdf, ax=ax2, palette=palette)
            ax2.set_title("Return Distribution by Label")
            ax2.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

        # 3. Statistics Panel
        ax3.axis("off")
        ax3.set_title("Statistics", fontweight="bold", pad=20)

        total = len(pdf)
        counts = pdf["label"].value_counts().sort_index()

        stat_lines = [
            f"Total ROI Samples: {total}",
            "-" * 25,
        ]

        # Label Counts & Percentages
        label_names = {-1: "Stop Loss", 0: "Vertical", 1: "Take Profit"}
        for label in [-1, 0, 1]:
            count = counts.get(label, 0)
            pct = (count / total * 100) if total > 0 else 0
            name = label_names.get(label, str(label))
            stat_lines.append(f"{name} ({label}): {count:5d} ({pct:5.1f}%)")

        stat_lines.append("-" * 25)

        # Vertical Barrier Rate
        vertical_rate = (counts.get(0, 0) / total * 100) if total > 0 else 0
        stat_lines.append(f"Vertical Barrier Rate: {vertical_rate:.1f}%")

        # Sampling Frequency (if t0 is present)
        if "t0" in pdf.columns:
            # Sort by t0 just in case
            pdf_sorted = pdf.sort_values("t0")
            intervals = pdf_sorted["t0"].diff().dropna()
            if not intervals.empty:
                avg_interval = intervals.mean()
                median_interval = intervals.median()
                stat_lines.append(f"Avg Sampling Interval: {avg_interval:.1f} bars")
                stat_lines.append(f"Med Sampling Interval: {median_interval:.1f} bars")

        # Add text to axis
        ax3.text(
            0.05,
            0.95,
            "\n\n".join(stat_lines),
            transform=ax3.transAxes,
            fontsize=12,
            family="monospace",
            verticalalignment="top",
        )

        self.save_plot(filename)

    def plot_feature_heatmap(
        self, features_df: pl.DataFrame, filename: str = "feature_heatmap.png"
    ):
        """
        Plot correlation heatmap of features.

        Args:
            features_df: Features DataFrame.
            filename: Output filename.
        """
        if features_df.is_empty():
            return

        # Exclude non-numeric or time columns
        cols = [
            c
            for c in features_df.columns
            if c not in ["datetime", "open", "high", "low", "close", "volume"]
        ]
        if not cols:
            return

        pdf = features_df.select(cols).to_pandas()
        corr = pdf.corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
        plt.title("Feature Correlation Matrix")
        self.save_plot(filename)

    def plot_stationarity_search(
        self,
        d_values: np.ndarray,
        p_values: np.ndarray,
        filename: str = "stationarity_search.png",
    ):
        """
        Plot ADF p-values vs Fractional Differentiation d.

        Args:
            d_values: Array of d values checked.
            p_values: Array of corresponding p-values.
            filename: Output filename.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(d_values, p_values, marker="o", linestyle="-")
        plt.axhline(y=0.05, color="r", linestyle="--", label="Critical Value (0.05)")
        plt.xlabel("Fractional Differentiation Order (d)")
        plt.ylabel("ADF p-value")
        plt.title("Stationarity Search: p-value vs d")
        plt.legend()
        plt.grid(True)
        self.save_plot(filename)

    def plot_cv_timeline(
        self, splits: List[Tuple], n_samples: int, filename: str = "cv_timeline.png"
    ):
        """
        Plot Cross-Validation Timeline (Purged K-Fold).

        Args:
            splits: List of (train_idx, test_idx).
            n_samples: Total number of samples.
            filename: Output filename.
        """
        n_splits = len(splits)
        fig, ax = plt.subplots(figsize=(12, n_splits * 0.8 + 1))

        # Create a grid for visualization (samples x splits)
        # 0 = embargo/unused, 1 = train, 2 = test

        for i, (train, test) in enumerate(splits):
            # Draw training
            ax.scatter(
                train,
                [i] * len(train),
                c="blue",
                marker="|",
                s=100,
                label="Train" if i == 0 else "",
            )
            # Draw test
            ax.scatter(
                test,
                [i] * len(test),
                c="red",
                marker="|",
                s=100,
                label="Test" if i == 0 else "",
            )

        ax.set_yticks(range(n_splits))
        ax.set_yticklabels([f"Fold {i + 1}" for i in range(n_splits)])
        ax.set_xlabel("Sample Index")
        ax.set_title("Purged K-Fold Cross-Validation Timeline")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        self.save_plot(filename)

    def plot_meta_performance(
        self,
        y_actual: np.ndarray,
        y_predicted: np.ndarray,
        filename: str = "meta_performance.png",
    ):
        """
        Plot detailed performance of Meta-Model.

        Args:
            y_actual: Actual labels (0/1).
            y_predicted: Predicted probabilities or labels.
            filename: Output filename.
        """
        from sklearn.metrics import (
            roc_curve,
            precision_recall_curve,
            auc,
            confusion_matrix,
        )

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Confusion Matrix (assuming y_predicted are labels or thresholded)
        y_pred_labels = (y_predicted > 0.5).astype(int)
        cm = confusion_matrix(y_actual, y_pred_labels)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
        axes[0].set_title("Confusion Matrix")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_actual, y_predicted)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        axes[1].plot([0, 1], [0, 1], "k--")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve")
        axes[1].legend()

        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_actual, y_predicted)
        pr_auc = auc(recall, precision)
        axes[2].plot(recall, precision, label=f"AUC = {pr_auc:.2f}")
        axes[2].set_xlabel("Recall")
        axes[2].set_ylabel("Precision")
        axes[2].set_title("Precision-Recall Curve")
        axes[2].legend()

        self.save_plot(filename)

    def plot_equity_curve(
        self, strategy_returns: np.ndarray, filename: str = "equity_curve.png"
    ):
        """
        Plot cumulative equity curve and drawdown.

        Args:
            strategy_returns: Array of strategy returns.
            filename: Output filename.
        """
        cumulative = np.cumsum(strategy_returns)

        # Calculate Drawdown
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )

        ax1.plot(cumulative, label="Strategy Equity")
        ax1.set_title("Strategy Performance")
        ax1.set_ylabel("Cumulative Returns")
        ax1.legend()

        ax2.fill_between(range(len(drawdown)), drawdown, 0, color="red", alpha=0.3)
        ax2.plot(drawdown, color="red", alpha=0.6)
        ax2.set_title("Drawdown")
        ax2.set_ylabel("Drawdown")
        ax2.set_xlabel("Trade/Bar Index")

        self.save_plot(filename)
