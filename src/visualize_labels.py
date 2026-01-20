"""
Visualize Triple Barrier Labels

This script creates comprehensive visualizations of the labeled data to:
1. Understand label distribution and quality
2. Analyze barrier touch patterns
3. Validate labeling effectiveness
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (16, 10)


def plot_label_distribution(events: pd.DataFrame):
    """Plot the distribution of labels."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Label counts
    label_counts = events["label"].value_counts().sort_index()
    label_names = {-1: "Loss", 0: "Neutral", 1: "Profit"}
    colors = {-1: "#e74c3c", 0: "#95a5a6", 1: "#2ecc71"}

    ax = axes[0, 0]
    bars = ax.bar(
        range(len(label_counts)),
        label_counts.values,
        color=[colors[k] for k in label_counts.index],
        alpha=0.7,
        edgecolor="black",
    )
    ax.set_xticks(range(len(label_counts)))
    ax.set_xticklabels([label_names[k] for k in label_counts.index])
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_title("Label Distribution (Counts)", fontweight="bold", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 2. Label percentages
    ax = axes[0, 1]
    label_pcts = events["label"].value_counts(normalize=True).sort_index() * 100
    bars = ax.bar(
        range(len(label_pcts)),
        label_pcts.values,
        color=[colors[k] for k in label_pcts.index],
        alpha=0.7,
        edgecolor="black",
    )
    ax.set_xticks(range(len(label_pcts)))
    ax.set_xticklabels([label_names[k] for k in label_pcts.index])
    ax.set_ylabel("Percentage (%)", fontweight="bold")
    ax.set_title("Label Distribution (Percentage)", fontweight="bold", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 3. Return distribution by label
    ax = axes[1, 0]
    for label in sorted(events["label"].unique()):
        returns = events[events["label"] == label]["ret"] * 100
        ax.hist(
            returns,
            bins=50,
            alpha=0.5,
            label=label_names[label],
            color=colors[label],
            edgecolor="black",
        )
    ax.set_xlabel("Return (%)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("Return Distribution by Label", fontweight="bold", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Box plot of returns by label
    ax = axes[1, 1]
    data_to_plot = [
        events[events["label"] == label]["ret"] * 100
        for label in sorted(events["label"].unique())
    ]
    bp = ax.boxplot(
        data_to_plot,
        labels=[label_names[k] for k in sorted(events["label"].unique())],
        patch_artist=True,
        widths=0.6,
    )

    # Color the boxes
    for patch, label in zip(bp["boxes"], sorted(events["label"].unique())):
        patch.set_facecolor(colors[label])
        patch.set_alpha(0.7)

    ax.set_ylabel("Return (%)", fontweight="bold")
    ax.set_title("Return Distribution (Box Plot)", fontweight="bold", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("label_distribution.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: label_distribution.png")
    plt.show()


def plot_temporal_analysis(events: pd.DataFrame):
    """Analyze labels over time."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # 1. Labels over time
    ax = axes[0]
    events_sorted = events.sort_index()
    colors_map = {-1: "#e74c3c", 0: "#95a5a6", 1: "#2ecc71"}

    for label in sorted(events["label"].unique()):
        mask = events_sorted["label"] == label
        ax.scatter(
            events_sorted[mask].index,
            events_sorted[mask]["ret"] * 100,
            c=colors_map[label],
            alpha=0.5,
            s=10,
            label={-1: "Loss", 0: "Neutral", 1: "Profit"}[label],
        )

    ax.set_ylabel("Return (%)", fontweight="bold")
    ax.set_title("Labels Over Time", fontweight="bold", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)

    # 2. Rolling label balance
    ax = axes[1]
    window = 100
    rolling_profit = (
        events_sorted["label"]
        .rolling(window)
        .apply(lambda x: (x == 1).sum() / len(x) * 100)
    )
    rolling_loss = (
        events_sorted["label"]
        .rolling(window)
        .apply(lambda x: (x == -1).sum() / len(x) * 100)
    )

    ax.plot(
        events_sorted.index,
        rolling_profit,
        label="Profit %",
        color="#2ecc71",
        linewidth=2,
    )
    ax.plot(
        events_sorted.index, rolling_loss, label="Loss %", color="#e74c3c", linewidth=2
    )
    ax.fill_between(
        events_sorted.index, rolling_profit, rolling_loss, alpha=0.2, color="gray"
    )

    ax.set_ylabel("Percentage (%)", fontweight="bold")
    ax.set_title(
        f"Rolling Label Balance (Window: {window} bars)", fontweight="bold", fontsize=14
    )
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(y=50, color="black", linestyle="--", alpha=0.3)

    # 3. Monthly label distribution
    ax = axes[2]
    events_monthly = events_sorted.copy()
    events_monthly["month"] = events_monthly.index.to_period("M")

    monthly_dist = (
        events_monthly.groupby(["month", "label"]).size().unstack(fill_value=0)
    )
    monthly_dist_pct = monthly_dist.div(monthly_dist.sum(axis=1), axis=0) * 100

    # Plot stacked bar chart
    months = [str(m) for m in monthly_dist_pct.index]
    x = np.arange(len(months))

    bottom = np.zeros(len(months))
    for label in sorted(monthly_dist_pct.columns):
        label_name = {-1: "Loss", 0: "Neutral", 1: "Profit"}[label]
        color = {-1: "#e74c3c", 0: "#95a5a6", 1: "#2ecc71"}[label]
        ax.bar(
            x,
            monthly_dist_pct[label],
            bottom=bottom,
            label=label_name,
            color=color,
            alpha=0.7,
            edgecolor="black",
        )
        bottom += monthly_dist_pct[label]

    ax.set_xticks(x[::3])  # Show every 3rd month
    ax.set_xticklabels(months[::3], rotation=45, ha="right")
    ax.set_ylabel("Percentage (%)", fontweight="bold")
    ax.set_title("Monthly Label Distribution", fontweight="bold", fontsize=14)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("temporal_analysis.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: temporal_analysis.png")
    plt.show()


def plot_barrier_analysis(events: pd.DataFrame):
    """Analyze barrier touch patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Holding period distribution
    ax = axes[0, 0]
    events_with_t1 = events.dropna(subset=["t1"]).copy()
    events_with_t1["t1"] = pd.to_datetime(events_with_t1["t1"])
    holding_periods = (
        events_with_t1["t1"] - events_with_t1.index
    ).dt.total_seconds() / 3600

    ax.hist(holding_periods, bins=50, alpha=0.7, color="#3498db", edgecolor="black")
    ax.set_xlabel("Holding Period (hours)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("Holding Period Distribution", fontweight="bold", fontsize=14)
    ax.grid(alpha=0.3)
    ax.axvline(
        holding_periods.median(),
        color="red",
        linestyle="--",
        label=f"Median: {holding_periods.median():.1f}h",
    )
    ax.legend()

    # 2. Holding period by label
    ax = axes[0, 1]
    for label in sorted(events["label"].unique()):
        mask = events_with_t1["label"] == label
        # Ensure 't1' is datetime within the loop context, though it's already converted above
        events_with_t1.loc[mask, "t1"] = pd.to_datetime(events_with_t1.loc[mask, "t1"])
        hp = (
            events_with_t1[mask]["t1"] - events_with_t1[mask].index
        ).dt.total_seconds() / 3600
        label_name = {-1: "Loss", 0: "Neutral", 1: "Profit"}[label]
        color = {-1: "#e74c3c", 0: "#95a5a6", 1: "#2ecc71"}[label]
        ax.hist(
            hp, bins=30, alpha=0.5, label=label_name, color=color, edgecolor="black"
        )

    ax.set_xlabel("Holding Period (hours)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("Holding Period by Label", fontweight="bold", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Target (volatility) distribution
    ax = axes[1, 0]
    ax.hist(
        events["trgt"] * 100, bins=50, alpha=0.7, color="#9b59b6", edgecolor="black"
    )
    ax.set_xlabel("Target Volatility (%)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title("Barrier Width Distribution", fontweight="bold", fontsize=14)
    ax.grid(alpha=0.3)
    ax.axvline(
        events["trgt"].median() * 100,
        color="red",
        linestyle="--",
        label=f"Median: {events['trgt'].median() * 100:.2f}%",
    )
    ax.legend()

    # 4. Return vs Target
    ax = axes[1, 1]
    colors_map = {-1: "#e74c3c", 0: "#95a5a6", 1: "#2ecc71"}
    for label in sorted(events["label"].unique()):
        mask = events["label"] == label
        label_name = {-1: "Loss", 0: "Neutral", 1: "Profit"}[label]
        ax.scatter(
            events[mask]["trgt"] * 100,
            events[mask]["ret"] * 100,
            c=colors_map[label],
            alpha=0.5,
            s=20,
            label=label_name,
        )

    ax.set_xlabel("Target Volatility (%)", fontweight="bold")
    ax.set_ylabel("Realized Return (%)", fontweight="bold")
    ax.set_title("Return vs Barrier Width", fontweight="bold", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("barrier_analysis.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: barrier_analysis.png")
    plt.show()


def print_summary_statistics(events: pd.DataFrame):
    """Print comprehensive summary statistics."""
    print("\n" + "=" * 80)
    print("TRIPLE BARRIER LABELING - SUMMARY STATISTICS")
    print("=" * 80)

    # Basic stats
    print(f"\n📊 Dataset Overview:")
    print(f"   Total events: {len(events):,}")
    print(f"   Date range: {events.index.min()} to {events.index.max()}")
    print(f"   Duration: {(events.index.max() - events.index.min()).days} days")

    # Label distribution
    print(f"\n🏷️  Label Distribution:")
    label_dist = events["label"].value_counts(normalize=True).sort_index()
    for label, pct in label_dist.items():
        label_name = {-1: "Loss", 0: "Neutral", 1: "Profit"}[label]
        print(
            f"   {label_name:>8}: {pct * 100:>5.2f}% ({events['label'].value_counts()[label]:>5,} events)"
        )

    # Return statistics
    print(f"\n💰 Return Statistics:")
    print(f"   Mean: {events['ret'].mean() * 100:>8.3f}%")
    print(f"   Median: {events['ret'].median() * 100:>8.3f}%")
    print(f"   Std: {events['ret'].std() * 100:>8.3f}%")
    print(f"   Min: {events['ret'].min() * 100:>8.3f}%")
    print(f"   Max: {events['ret'].max() * 100:>8.3f}%")

    # Return by label
    print(f"\n📈 Return by Label:")
    for label in sorted(events["label"].unique()):
        label_name = {-1: "Loss", 0: "Neutral", 1: "Profit"}[label]
        rets = events[events["label"] == label]["ret"] * 100
        print(f"   {label_name:>8}: Mean={rets.mean():>7.3f}%, Std={rets.std():>7.3f}%")

    # Holding period
    events_with_t1 = events.dropna(subset=["t1"])
    events_with_t1["t1"] = pd.to_datetime(events_with_t1["t1"])
    holding_periods = (
        events_with_t1["t1"] - events_with_t1.index
    ).dt.total_seconds() / 3600
    print(f"\n⏱️  Holding Period (hours):")
    print(f"   Mean: {holding_periods.mean():>8.2f}")
    print(f"   Median: {holding_periods.median():>8.2f}")
    print(f"   Min: {holding_periods.min():>8.2f}")
    print(f"   Max: {holding_periods.max():>8.2f}")

    # Barrier width
    print(f"\n🎯 Barrier Width (volatility):")
    print(f"   Mean: {events['trgt'].mean() * 100:>8.3f}%")
    print(f"   Median: {events['trgt'].median() * 100:>8.3f}%")
    print(f"   Min: {events['trgt'].min() * 100:>8.3f}%")
    print(f"   Max: {events['trgt'].max() * 100:>8.3f}%")

    print("\n" + "=" * 80)


def main():
    """Main visualization workflow."""
    print("=" * 80)
    print("Triple Barrier Label Visualization")
    print("=" * 80)

    # Load labeled events
    print("\nLoading labeled events...")
    events = pd.read_csv("labeled_events.csv", index_col=0, parse_dates=True)
    print(f"✓ Loaded {len(events)} events")

    # Print summary
    print_summary_statistics(events)

    # Generate visualizations
    print("\n" + "=" * 80)
    print("Generating Visualizations...")
    print("=" * 80)

    print("\n1. Label Distribution Analysis...")
    plot_label_distribution(events)

    print("\n2. Temporal Analysis...")
    plot_temporal_analysis(events)

    print("\n3. Barrier Analysis...")
    plot_barrier_analysis(events)

    print("\n" + "=" * 80)
    print("✓ Visualization Complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - label_distribution.png")
    print("  - temporal_analysis.png")
    print("  - barrier_analysis.png")


if __name__ == "__main__":
    main()
