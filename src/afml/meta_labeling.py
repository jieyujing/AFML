"""
Meta-Labeling Pipeline for Financial Machine Learning (Polars Optimized).

This module implements Meta-Labeling using Polars for improved
performance on large-scale financial time series data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
from polars import DataFrame, LazyFrame, Series

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MetaLabelingPipeline:
    """
    Meta-Labeling Pipeline for filtering primary model predictions.

    Two-layer approach:
    1. Primary Model: Predicts direction (long/short)
    2. Meta Model: Filters primary predictions to improve precision

    This implementation uses Polars for improved performance on large datasets.

    Attributes:
        primary_model: sklearn-compatible model for direction prediction
        meta_model: sklearn-compatible model for meta-labeling
        primary_params: Parameters for primary model
        meta_params: Parameters for meta model
        n_splits: Number of cross-validation splits (default 5)

    Example:
        >>> pipeline = MetaLabelingPipeline()
        >>> pipeline.fit(X, y)
        >>> predictions = pipeline.predict(X)
    """

    def __init__(
        self,
        primary_model: str = "random_forest",
        meta_model: str = "logistic",
        primary_params: Optional[Dict] = None,
        meta_params: Optional[Dict] = None,
        n_splits: int = 5,
        embargo: float = 0.01,
        *,
        random_state: int = 42,
    ):
        """
        Initialize MetaLabelingPipeline.

        Args:
            primary_model: Type of primary model ('random_forest', 'lr')
            meta_model: Type of meta model ('random_forest', 'lr')
            primary_params: Parameters for primary model
            meta_params: Parameters for meta model
            n_splits: Number of cross-validation splits
            embargo: Embargo proportion
            random_state: Random seed
        """
        self.primary_model = primary_model
        self.meta_model = meta_model
        self.n_splits = n_splits
        self.embargo = embargo
        self.cv_ = None
        self.primary_probs_ = None
        self.primary_params = primary_params or {
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": random_state,
        }
        self.meta_params = meta_params or {
            "C": 1.0,
            "random_state": random_state,
        }
        self.random_state = random_state

        self._primary_clf = self._create_primary_model()
        self._meta_clf = self._create_meta_model()

        self._is_fitted = False
        self._feature_names: Optional[List[str]] = None

    def _create_primary_model(self) -> Any:
        """Create primary classifier."""
        if self.primary_model == "random_forest":
            return RandomForestClassifier(**self.primary_params)
        elif self.primary_model == "lr":
            return LogisticRegression(**self.meta_params)
        else:
            return RandomForestClassifier(**self.primary_params)

    def _create_meta_model(self) -> Any:
        """Create meta classifier."""
        return HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=5,
            random_state=self.random_state,
        )

    def fit(
        self,
        X: Union[DataFrame, LazyFrame, np.ndarray],
        y: Union[Series, np.ndarray],
        *,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "MetaLabelingPipeline":
        """
        Fit primary and meta models.
        """
        X_array, feature_names = self._prepare_features(X)

        if isinstance(y, (DataFrame, LazyFrame)):
            if "label" in y.columns:
                y = y["label"]
            y = y.to_numpy()
        elif isinstance(y, Series):
            y = y.to_numpy()

        # 1. Fit primary model
        self._primary_clf.fit(X_array, y, sample_weight=sample_weight)

        # 2. Get primary predictions and confidence
        primary_pred = self._primary_clf.predict(X_array)
        primary_proba_all = self._primary_clf.predict_proba(X_array)
        primary_confidence = np.max(primary_proba_all, axis=1)

        # 3. Define Meta-Labeling target
        y_meta = ((primary_pred == y) & (primary_pred != 0)).astype(int)

        # 4. Fit meta model
        meta_features = np.column_stack([X_array, primary_confidence])
        self._meta_clf.fit(meta_features, y_meta, sample_weight=sample_weight)

        self._is_fitted = True
        self._feature_names = feature_names

        return self

    def _prepare_features(
        self,
        X: Union[DataFrame, LazyFrame, np.ndarray],
    ) -> tuple[np.ndarray, Optional[List[str]]]:
        """Prepare features for sklearn models."""
        if isinstance(X, (DataFrame, LazyFrame)):
            feature_names = X.columns
            if isinstance(X, LazyFrame):
                X = X.collect()
            X_array = X.to_numpy()
        else:
            feature_names = None
            X_array = np.array(X)

        return X_array, feature_names

    def predict(
        self,
        X: Union[DataFrame, LazyFrame, np.ndarray],
    ) -> np.ndarray:
        """Generate binary predictions using meta-labeling."""
        if not self._is_fitted:
            raise ValueError("Pipeline must be fitted before predict")

        X_array, _ = self._prepare_features(X)

        primary_pred = self._primary_clf.predict(X_array)
        primary_proba_all = self._primary_clf.predict_proba(X_array)
        primary_confidence = np.max(primary_proba_all, axis=1)

        meta_features = np.column_stack([X_array, primary_confidence])
        meta_pred = self._meta_clf.predict(meta_features)

        # Signal filter
        final_pred = primary_pred * meta_pred

        return final_pred

    def score(
        self,
        X: Union[DataFrame, LazyFrame, np.ndarray],
        y: Union[Series, np.ndarray],
        *,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        if isinstance(y, (DataFrame, LazyFrame)):
            if "label" in y.columns:
                y = y["label"]
            y = y.to_numpy()
        elif isinstance(y, Series):
            y = y.to_numpy()

        y_binary = (y > 0).astype(int)
        predictions = self.predict(X)
        predictions_binary = (predictions > 0).astype(int)

        return {
            "accuracy": accuracy_score(y_binary, predictions_binary),
            "precision": precision_score(y_binary, predictions_binary, zero_division=0),
            "recall": recall_score(y_binary, predictions_binary, zero_division=0),
            "f1": f1_score(y_binary, predictions_binary, zero_division=0),
        }

    def _sharpe_ratio(self, returns: np.ndarray, risk_free: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free
        if np.std(excess_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def _sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-10
        return np.mean(returns) / downside_std * np.sqrt(252)

    def _max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = peak - cumulative_returns
        return np.max(drawdown)
