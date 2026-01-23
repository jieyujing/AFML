import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import sys

# Ensure src is in path imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def explain_pca_components():
    print("Loading data for PCA explanation...")
    df = pd.read_csv(os.path.join("data", "output", "features_v2_labeled.csv"))
    
    # --- Replicate the exact preprocessing from feature_pca.py ---
    exclude_cols = ['bin', 'w', 'avg_u', 'ret', 'close', 'date', 'sample_weight', 'label', 'return', 'holding_period']
    unnamed_cols = [c for c in df.columns if 'Unnamed' in c]
    
    X = df.select_dtypes(include=[np.number]).copy()
    X = X.drop(columns=[c for c in exclude_cols + unnamed_cols if c in X.columns])
    
    if X.isnull().values.any():
        X = X.ffill().fillna(0)
    
    feature_names = X.columns.tolist()
    
    # --- Replicate PCA Fit ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=0.95)
    pca.fit(X_scaled)
    
    n_pcs = pca.n_components_
    component_names = [f'PC_{i+1}' for i in range(n_pcs)]
    
    print(f"PCA Fitted: {n_pcs} components explain 95% variance.")
    
    # --- Analyze Loadings ---
    # Loadings matrix: Rows = PCs, Cols = Original Features
    loadings = pd.DataFrame(
        pca.components_, 
        columns=feature_names, 
        index=component_names
    )
    
    # 1. Identify Key Components from previous LightGBM results
    # Ideally adapt this list based on feature_importance results
    # From PROGRESS.md (Path 19): Top features were PC_9, PC_1, PC_31
    target_pcs = ['PC_1', 'PC_9', 'PC_31']
    # Check if they exist in current model (might differ if data changed slightly, but assuming consisteny)
    target_pcs = [pc for pc in target_pcs if pc in loadings.index]
    
    # Add top 2 variance explanations just in case
    if 'PC_1' not in target_pcs: target_pcs.insert(0, 'PC_1')
    if 'PC_2' not in target_pcs: target_pcs.append('PC_2')
    
    print("\n--- Component Interpretation ---")
    
    analysis_results = []
    
    for pc in target_pcs:
        print(f"\nAnalyzing {pc} (Explained Var: {pca.explained_variance_ratio_[int(pc.split('_')[1])-1]:.2%}):")
        
        # Get top 5 absolute loadings
        pc_loadings = loadings.loc[pc].sort_values(key=abs, ascending=False)
        top_5 = pc_loadings.head(5)
        
        desc_str = []
        for feat, val in top_5.items():
            sign = "+" if val > 0 else "-"
            print(f"  {sign} {feat} ({val:.4f})")
            desc_str.append(f"{sign}{feat}")
            
        analysis_results.append({
            'Component': pc,
            'Variance': pca.explained_variance_ratio_[int(pc.split('_')[1])-1],
            'Top Definition': ", ".join(desc_str)
        })

    # 2. Visualization: Heatmap of Top Loadings
    # Select subset of features that are important across these components
    important_feats = set()
    for pc in target_pcs:
        important_feats.update(loadings.loc[pc].sort_values(key=abs, ascending=False).head(5).index.tolist())
    
    subset_loadings = loadings.loc[target_pcs, list(important_feats)]
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(subset_loadings, cmap='RdBu', center=0, annot=True, fmt='.2f')
    plt.title('PCA Loadings Interpretation (Key Components)')
    plt.tight_layout()
    os.makedirs('visual_analysis', exist_ok=True)
    plt.savefig('visual_analysis/pca_loadings_heatmap.png')
    print("Saved PCA loadings heatmap.")

    # Save detailed loadings
    loadings.to_csv(os.path.join("data", "output", "pca_loadings_matrix.csv"))
    pd.DataFrame(analysis_results).to_csv(os.path.join("data", "output", "pca_component_definitions.csv"), index=False)

if __name__ == "__main__":
    explain_pca_components()
