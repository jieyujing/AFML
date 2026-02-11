"""
Base classes and mixins for AFML processors.

This module provides abstract base classes that ensure consistent interface
across all processor classes in the AFML library.
"""

from abc import ABC, abstractmethod
from typing import Any


class ProcessorMixin(ABC):
    """
    Abstract base class for all AFML processor classes.

    This class defines the standard interface that all processors should follow:
    - fit(): Learn parameters from data
    - transform(): Apply learned parameters to transform data
    - fit_transform(): Convenience method combining fit and transform

    All processors are designed to be sklearn-compatible for use in pipelines.
    """

    def __init__(self):
        """Initialize the processor."""
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """
        Check if the processor has been fitted.

        Returns:
            bool: True if fit() has been called, False otherwise
        """
        return self._is_fitted

    @abstractmethod
    def fit(self, X: Any, y: Any = None) -> "ProcessorMixin":
        """
        Learn parameters from data.

        Args:
            X: Input data (typically pandas DataFrame)
            y: Optional target data (typically pandas Series)

        Returns:
            self: The fitted processor instance
        """
        self._is_fitted = True
        return self

    @abstractmethod
    def transform(self, X: Any) -> Any:
        """
        Apply learned parameters to transform data.

        Args:
            X: Input data to transform

        Returns:
            Transformed data
        """
        raise NotImplementedError

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        """
        Learn parameters from data and transform it.

        This is a convenience method that calls fit() followed by transform().
        It is compatible with sklearn's Pipeline and ColumnTransformer.

        Args:
            X: Input data
            y: Optional target data

        Returns:
            Transformed data
        """
        self.fit(X, y)
        return self.transform(X)

    def __repr__(self) -> str:
        """Return a string representation of the processor."""
        attrs = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }
        attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items())
        return f"{self.__class__.__name__}({attr_str})"


class ConfigurableProcessorMixin(ProcessorMixin):
    """
    Extended base class for processors that load configuration from YAML files.

    This mixin provides automatic loading of default configuration from
    config/processor_defaults.yaml and supports runtime parameter overrides.
    """

    config_file: str = "config/processor_defaults.yaml"
    config_section: str = ""

    def __init__(self, **kwargs):
        """
        Initialize the processor with configuration.

        Args:
            **kwargs: Runtime parameters that override config file values
        """
        super().__init__()
        self.params = self._load_config()
        self.params.update(kwargs)
        self._apply_params()

    def _load_config(self) -> dict:
        """
        Load default configuration from YAML file.

        Returns:
            Dictionary containing default configuration values
        """
        import os

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            self.config_file,
        )

        try:
            import yaml

            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if config and self.config_section:
                    return config.get(self.config_section, {})
                return config or {}
        except FileNotFoundError:
            return {}
        except Exception:
            return {}

    def _apply_params(self) -> None:
        """
        Apply parameters to instance attributes.

        Override this method in subclasses to handle specific parameters.
        """
        pass

    def get_params(self) -> dict:
        """
        Get all current parameters.

        Returns:
            Dictionary of current parameter values
        """
        return self.params.copy()

    def set_params(self, **kwargs) -> "ConfigurableProcessorMixin":
        """
        Set parameters after initialization.

        Args:
            **kwargs: Parameters to update

        Returns:
            self
        """
        self.params.update(kwargs)
        self._apply_params()
        return self
