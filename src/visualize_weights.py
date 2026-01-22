"""
Visualize Sample Weights and Uniqueness
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    print("=" * 80)
    print("Visualizing Sample Weights")
    print("=" * 80)

    # 1. Load Data
    try:
        df = pd.read_csv("sample_weights.csv", index_col=0, parse_dates=True)
        print(f"Loaded {len(df)} weighted events")
    except FileNotFoundError:
        print("Error: sample_weights.csv not found.")
        return

    # Create visualization directory
    os.makedirs("visual_analysis", exist_ok=True)

    # Set style
    sns.set_theme(style="whitegrid")
    
    # Figure 1: Distribution of Uniqueness and Weights
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Uniqueness Histogram
    sns.histplot(data=df, x="avg_uniqueness", bins=50, kde=True, ax=axes[0, 0], color="skyblue")
    axes[0, 0].set_title("Distribution of Average Uniqueness")
    axes[0, 0].set_xlabel("Average Uniqueness")
    axes[0, 0].axvline(df["avg_uniqueness"].mean(), color="red", linestyle="--", label=f"Mean: {df['avg_uniqueness'].mean():.2f}")
    axes[0, 0].legend()

    # Weights Histogram
    sns.histplot(data=df, x="sample_weight", bins=50, kde=True, ax=axes[0, 1], color="orange")
    axes[0, 1].set_title("Distribution of Sample Weights")
    axes[0, 1].set_xlabel("Sample Weight")
    axes[0, 1].axvline(df["sample_weight"].mean(), color="red", linestyle="--", label=f"Mean: {df['sample_weight'].mean():.2f}")
    axes[0, 1].legend()

    # Uniqueness over Time
    df["avg_uniqueness"].plot(ax=axes[1, 0], alpha=0.6, color="blue")
    axes[1, 0].set_title("Average Uniqueness over Time")
    axes[1, 0].set_ylabel("Uniqueness")

    # Weights over Time
    df["sample_weight"].plot(ax=axes[1, 1], alpha=0.6, color="green")
    axes[1, 1].set_title("Sample Weights over Time")
    axes[1, 1].set_ylabel("Weight")

    plt.tight_layout()
    plt.savefig("visual_analysis/weights_distribution.png")
    print("   ✓ Saved weights_distribution.png")
    plt.close()

    # Figure 2: Weights vs Returns and Concurrency
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Weight vs Absolute Return
    sns.scatterplot(data=df, x=df["ret"].abs(), y="sample_weight", alpha=0.5, ax=axes[0])
    axes[0].set_title("Sample Weight vs Absolute Return")
    axes[0].set_xlabel("Absolute Return")
    axes[0].set_ylabel("Sample Weight")
    
    # Weight vs Uniqueness
    sns.scatterplot(data=df, x="avg_uniqueness", y="sample_weight", alpha=0.5, ax=axes[1])
    axes[1].set_title("Sample Weight vs Average Uniqueness")
    axes[1].set_xlabel("Average Uniqueness")
    axes[1].set_ylabel("Sample Weight")

    plt.tight_layout()
    plt.savefig("visual_analysis/weights_correlations.png")
    print("   ✓ Saved weights_correlations.png")
    plt.close()

    print("\nVisualization Complete!")

if __name__ == "__main__":
    main()
