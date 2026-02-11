"""
Polars Meta-Labeling Pipeline for Financial Machine Learning.

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


class PolarsMetaLabelingPipeline:
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

    Example:
        >>> pipeline = PolarsMetaLabelingPipeline()
        >>> pipeline.fit(X, y)
        >>> predictions = pipeline.predict(X)
    """

    def __init__(
        self,
        primary_model: str = "random_forest",
        meta_model: str = "logistic",
        primary_params: Optional[Dict] = None,
        meta_params: Optional[Dict] = None,
        *,
        random_state: int = 42,
    ):
        """
        Initialize PolarsMetaLabelingPipeline.

        Args:
            primary_model: Type of primary model ('random_forest', 'lr')
            meta_model: Type of meta model ('random_forest', 'lr')
            primary_params: Parameters for primary model
            meta_params: Parameters for meta model
            random_state: Random seed
        """
        self.primary_model = primary_model
        self.meta_model = meta_model
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
        if self.meta_model == "random_forest":
            return RandomForestClassifier(**self.primary_params)
        elif self.meta_model == "lr":
            # Use HistGradientBoostingClassifier for NaN tolerance
            return HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=5,
                random_state=self.random_state,
            )
        else:
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
    ) -> "PolarsMetaLabelingPipeline":
        """
        Fit primary and meta models.

        Args:
            X: Features
            y: Labels
            sample_weight: Optional sample weights

        Returns:
            self
        """
        X_array, feature_names = self._prepare_features(X)

        if isinstance(y, (DataFrame, LazyFrame)):
            if "label" in y.columns:
                y = y["label"]
            y = y.to_numpy()

        y_binary = (y > 0).astype(int)

        self._primary_clf.fit(X_array, y, sample_weight=sample_weight)

        primary_proba = self._primary_clf.predict_proba(X_array)[:, 1]

        meta_features = np.column_stack([X_array, primary_proba])
        self._meta_clf.fit(meta_features, y_binary, sample_weight=sample_weight)

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
        """
        Generate binary predictions using meta-labeling.

        Args:
            X: Features

        Returns:
            Binary predictions
        """
        if not self._is_fitted:
            raise ValueError("Pipeline must be fitted before predict")

        X_array, _ = self._prepare_features(X)

        primary_pred = self._primary_clf.predict(X_array)
        primary_proba = self._primary_clf.predict_proba(X_array)[:, 1]

        meta_features = np.column_stack([X_array, primary_proba])
        meta_pred = self._meta_clf.predict(meta_features)

        final_pred = primary_pred * meta_pred

        return final_pred

    def predict_proba(
        self,
        X: Union[DataFrame, LazyFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Features

        Returns:
            Probability estimates
        """
        if not self._is_fitted:
            raise ValueError("Pipeline must be fitted before predict_proba")

        X_array, _ = self._prepare_features(X)

        primary_proba = self._primary_clf.predict_proba(X_array)[:, 1]
        meta_proba = self._meta_clf.predict_proba(X_array)[:, 1]

        combined_proba = np.column_stack([primary_proba, meta_proba])

        return combined_proba

    def score(
        self,
        X: Union[DataFrame, LazyFrame, np.ndarray],
        y: Union[Series, np.ndarray],
        *,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Features
            y: True labels
            sample_weight: Optional sample weights

        Returns:
            Dict with evaluation metrics
        """
        if isinstance(y, (DataFrame, LazyFrame)):
            if "label" in y.columns:
                y = y["label"]
            y = y.to_numpy()

        y_binary = (y > 0).astype(int)
        predictions = self.predict(X)

        # Handle potential -1 values in predictions by converting to binary (1 for profit signal, 0 otherwise)
        predictions_binary = (predictions > 0).astype(int)

        return {
            "accuracy": accuracy_score(y_binary, predictions_binary),
            "precision": precision_score(y_binary, predictions_binary, zero_division=0),
            "recall": recall_score(y_binary, predictions_binary, zero_division=0),
            "f1": f1_score(y_binary, predictions_binary, zero_division=0),
        }

    def get_metrics(
        self,
        returns: Union[Series, np.ndarray],
        predictions: Union[Series, np.ndarray],
    ) -> Dict[str, float]:
        """
        Calculate performance metrics from returns and predictions.

        Args:
            returns: Strategy returns
            predictions: Trading signals

        Returns:
            Dict with performance metrics
        """
        if isinstance(returns, Series):
            returns = returns.to_numpy()
        if isinstance(predictions, Series):
            predictions = predictions.to_numpy()

        strategy_returns = returns * predictions

        sharpe = self._sharpe_ratio(strategy_returns)
        sortino = self._sortino_ratio(strategy_returns)

        cumulative = np.cumsum(strategy_returns)
        max_drawdown = self._max_drawdown(cumulative)

        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "total_return": cumulative[-1] if len(cumulative) > 0 else 0,
            "win_rate": (strategy_returns > 0).mean(),
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

    def cross_validate(
        self,
        X: Union[DataFrame, LazyFrame, np.ndarray],
        y: Union[Series, np.ndarray],
        cv: Any,
        *,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Perform cross-validation.

        Args:
            X: Features
            y: Labels
            cv: Cross-validation splitter
            sample_weight: Optional sample weights

        Returns:
            Dict with CV metrics
        """
        from sklearn.model_selection import cross_val_score

        X_array, _ = self._prepare_features(X)

        if isinstance(y, (DataFrame, LazyFrame)):
            if "label" in y.columns:
                y = y["label"]
            y = y.to_numpy()

        scores = cross_val_score(self, X_array, y, cv=cv, scoring="accuracy")

        return {
            "cv_accuracy_mean": scores.mean(),
            "cv_accuracy_std": scores.std(),
        }

    def fit_transform(
        self,
        X: Union[DataFrame, LazyFrame, np.ndarray],
        y: Union[Series, np.ndarray],
        *,
        sample_weight: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fit and predict in one step.

        Args:
            X: Features
            y: Labels
            sample_weight: Optional sample weights

        Returns:
            Binary predictions
        """
        self.fit(X, y, sample_weight=sample_weight)
        return self.predict(X)

    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline.

        Returns:
            Dict with pipeline configuration
        """
        return {
            "primary_model": self.primary_model,
            "meta_model": self.meta_model,
            "primary_params": self.primary_params,
            "meta_params": self.meta_params,
            "is_fitted": self._is_fitted,
            "feature_names": self._feature_names,
        }
