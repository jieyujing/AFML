"""
Meta-Labeling Pipeline for Financial Machine Learning.

This module implements the Meta-Labeling workflow from AFML Chapter 3 & 4,
orchestrating primary and secondary models for filtered predictions.
"""

import pandas as pd
from typing import Optional, List
from sklearn.ensemble import RandomForestClassifier

from .base import ProcessorMixin
from .cv import PurgedKFoldCV


class MetaLabelingPipeline(ProcessorMixin):
    """
    Orchestrates meta-labeling workflow.

    Trains a primary model for direction prediction, then uses a secondary
    model to filter the primary model's signals.

    Attributes:
        primary_model: Primary classifier
        secondary_model: Meta-label classifier
        primary_params: Hyperparameters for primary model
        secondary_params: Hyperparameters for secondary model
        n_splits: Number of CV splits
        embargo: Embargo percentage for PurgedKFold

    Example:
        >>> pipeline = MetaLabelingPipeline(n_splits=5)
        >>> pipeline.fit(X, y)
        >>> predictions = pipeline.predict(X_new)
    """

    def __init__(
        self,
        primary_model=None,
        secondary_model=None,
        n_splits: int = 5,
        embargo: float = 0.01,
    ):
        """
        Initialize the MetaLabelingPipeline.

        Args:
            primary_model: Primary classifier instance
            secondary_model: Secondary classifier instance
            n_splits: Number of PurgedKFold splits
            embargo: Embargo percentage
        """
        super().__init__()
        self.primary_model = (
            primary_model
            if primary_model is not None
            else RandomForestClassifier(n_estimators=100, random_state=42)
        )
        self.secondary_model = (
            secondary_model
            if secondary_model is not None
            else RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        )
        self.n_splits = n_splits
        self.embargo = embargo
        self.cv_ = None
        self.primary_probs_: Optional[pd.DataFrame] = None
        self.meta_clf_ = None

    def fit(
        self, df: pd.DataFrame, features: List[str], y: pd.Series
    ) -> "MetaLabelingPipeline":
        """
        Train both primary and secondary models.

        Args:
            df: DataFrame with features and metadata
            features: List of feature column names
            y: Labels

        Returns:
            self
        """
        X = df[features]
        weights = df.get("sample_weight")

        # Setup PurgedKFold
        self.cv_ = PurgedKFoldCV(
            n_splits=self.n_splits,
            samples_info_sets=df["t1"],
            embargo=self.embargo,
        )

        # Train primary model with OOS predictions
        primary_preds = pd.Series(0, index=df.index, name="primary_pred")
        self.primary_probs_ = pd.DataFrame(0.0, index=df.index, columns=[-1, 0, 1])
        tested_indices = []

        for train_idx, test_idx in self.cv_.split(X):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            w_train = weights.iloc[train_idx] if weights is not None else None

            self.primary_model.fit(X_train, y_train, sample_weight=w_train)

            preds = self.primary_model.predict(X_test)
            probs = self.primary_model.predict_proba(X_test)

            primary_preds.iloc[test_idx] = preds
            classes = self.primary_model.classes_

            for j, cls in enumerate(classes):
                if cls in [-1, 0, 1]:
                    self.primary_probs_.loc[df.index[test_idx], cls] = probs[:, j]

            tested_indices.extend(test_idx)

        # Generate meta-labels and train secondary model
        df_meta = df.iloc[tested_indices].copy()
        primary_preds = primary_preds.iloc[tested_indices]
        meta_labels = self._generate_meta_labels(
            df_meta, primary_preds, y.iloc[tested_indices]
        )

        # Train secondary model with enhanced features
        X_meta = df_meta.loc[meta_labels.index, features].copy()
        X_meta = self._add_confidence_feature(
            X_meta, primary_preds, self.primary_probs_
        )

        self.meta_clf_ = self.secondary_model
        self.meta_clf_.fit(X_meta, meta_labels)

        return self

    def _generate_meta_labels(
        self,
        df: pd.DataFrame,
        primary_preds: pd.Series,
        y_true: pd.Series,
    ) -> pd.Series:
        """
        Generate meta-labels from primary predictions.

        Args:
            df: DataFrame
            primary_preds: Primary model predictions
            y_true: True labels

        Returns:
            Meta-labels
        """
        mask = primary_preds != 0
        meta_labels = (primary_preds[mask] == y_true[mask]).astype(int)
        return meta_labels

    def _add_confidence_feature(
        self,
        X: pd.DataFrame,
        preds: pd.Series,
        probs: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add primary model confidence as feature."""
        confidences = []
        for idx in X.index:
            pred_class = preds.loc[idx]
            conf = probs.loc[idx, pred_class]
            confidences.append(conf)

        X["primary_model_prob"] = confidences
        return X

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Return filtered predictions from meta-model.

        Args:
            X: Feature DataFrame

        Returns:
            Filtered predictions
        """
        if self.meta_clf_ is None:
            raise ValueError("Pipeline has not been fitted.")

        # Get primary predictions
        primary_preds = self.primary_model.predict(X)

        # Get meta probabilities
        X_meta = X.copy()
        probs = self.primary_model.predict_proba(X)
        classes = self.primary_model.classes_

        confidences = []
        for i, idx in enumerate(X.index):
            pred_class = primary_preds[i]
            if pred_class in classes:
                cls_idx = list(classes).index(pred_class)
                confidences.append(probs[i, cls_idx])
            else:
                confidences.append(0.5)

        X_meta["primary_model_prob"] = confidences

        # Get meta predictions
        meta_preds = self.meta_clf_.predict(X_meta)

        # Filter primary predictions
        filtered = primary_preds.copy()
        for i, idx in enumerate(X.index):
            if meta_preds[i] == 0:
                filtered.iloc[i] = 0

        return filtered

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return prediction probabilities.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with probabilities
        """
        if self.meta_clf_ is None:
            raise ValueError("Pipeline has not been fitted.")

        X_meta = X.copy()
        probs = self.primary_model.predict_proba(X)
        classes = self.primary_model.classes_

        confidences = []
        for i, idx in enumerate(X.index):
            pred_class = self.primary_model.predict(X.iloc[[i]])[0]
            if pred_class in classes:
                cls_idx = list(classes).index(pred_class)
                confidences.append(probs[i, cls_idx])
            else:
                confidences.append(0.5)

        X_meta["primary_model_prob"] = confidences

        return pd.DataFrame(
            self.meta_clf_.predict_proba(X_meta),
            index=X.index,
            columns=self.meta_clf_.classes_,
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform is not applicable for pipeline.

        Use predict() or predict_proba() instead.
        """
        raise NotImplementedError("Use predict() or predict_proba() methods.")

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Fit and predict in one step.

        Args:
            X: Feature DataFrame
            y: Labels

        Returns:
            Predictions
        """
        features = [c for c in X.columns if c not in ["t1", "label", "ret"]]
        self.fit(X, features, y)
        return self.predict(X)
