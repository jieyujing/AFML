import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

class PCAFeatureReducer:
    """
    PCA Feature Reduction component following AFML and mlfinlab principles.
    Focuses on feature orthogonalization and dimensionality reduction.
    """
    
    def __init__(self, n_components=0.95):
        """
        Initialize PCA reducer.
        
        :param n_components: If float between 0 and 1, it represents the variance ratio to preserve.
                             If int, it represents the absolute number of components.
        """
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components)
        self.feature_names = None
        self.component_names = None

    def fit_transform(self, X):
        """
        Fit the scaler and PCA on X, then transform X.
        
        :param X: (pd.DataFrame) Feature matrix.
        :return: (pd.DataFrame) Reduced feature matrix.
        """
        self.feature_names = X.columns.tolist()
        
        # 1. Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # 2. Apply PCA
        X_pca = self.pca.fit_transform(X_scaled)
        
        # 3. Create component names
        n_pcs = X_pca.shape[1]
        self.component_names = [f'PC_{i+1}' for i in range(n_pcs)]
        
        return pd.DataFrame(X_pca, columns=self.component_names, index=X.index)

    def plot_variance(self, output_path='visual_analysis/pca_variance.png'):
        """
        Plot explained variance ratio.
        """
        plt.figure(figsize=(10, 6))
        exp_var = self.pca.explained_variance_ratio_
        cum_var = np.cumsum(exp_var)
        
        plt.bar(range(1, len(exp_var) + 1), exp_var, alpha=0.5, align='center', label='Individual variance')
        plt.step(range(1, len(cum_var) + 1), cum_var, where='mid', label='Cumulative variance')
        
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal component index')
        plt.legend(loc='best')
        plt.title('PCA Explained Variance Analysis')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print(f"PCA variance plot saved to {output_path}")

def run_pca_pipeline():
    """
    Main pipeline for PCA reduction.
    """
    print("Loading features...")
    # Load the latest feature set
    # Load the latest feature set
    df = pd.read_csv(os.path.join('data', 'output', 'features_v2_labeled.csv'))
    
    # 1. Drop non-numeric columns and explicit metadata
    exclude_cols = ['bin', 'w', 'avg_u', 'ret', 'close', 'date', 'sample_weight', 'label', 'return', 'holding_period']
    
    # Also drop index columns if they exist (e.g., Unnamed: 0)
    unnamed_cols = [c for c in df.columns if 'Unnamed' in c]
    
    # Select only numeric columns for PCA
    X = df.select_dtypes(include=[np.number]).copy()
    
    # Drop target and metadata from X
    X = X.drop(columns=[c for c in exclude_cols + unnamed_cols if c in X.columns])
    
    print(f"Original feature count: {X.shape[1]}")
    
    # Handle NaNs (required for PCA)
    if X.isnull().values.any():
        print("Handling NaNs using forward fill then zero fill...")
        X = X.ffill().fillna(0)
    
    # Apply PCA
    reducer = PCAFeatureReducer(n_components=0.95)
    X_pca = reducer.fit_transform(X)
    
    print(f"Reduced feature count (95% variance): {X_pca.shape[1]}")
    
    # Keep metadata for the final dataframe
    meta_cols = [c for c in df.columns if c in exclude_cols or c == 'date' or 'Unnamed' in c]
    df_pca = pd.concat([df[meta_cols], X_pca], axis=1)
    
    # Save results
    output_dir = os.path.join('data', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'features_pca.csv')
    df_pca.to_csv(output_file, index=False)
    print(f"PCA features saved to {output_file}")
    
    # Visual analysis
    reducer.plot_variance()

if __name__ == "__main__":
    run_pca_pipeline()
