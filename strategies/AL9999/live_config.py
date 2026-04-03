"""
AL9999 vn.py live trading configuration helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from strategies.AL9999.config import (
    CUSUM_MULTIPLIER,
    CUSUM_WINDOW,
    EWMA_SPAN,
    FEATURE_CONFIG,
    FRACDIFF_THRES,
    MA_PRIMARY_MODEL,
    META_MODEL_CONFIG,
    TBM_CONFIG,
    TARGET_DAILY_BARS,
)


@dataclass
class Al9999LiveConfig:
    """
    Runtime configuration for the AL9999 live adapter.
    """

    research_symbol: str = "AL9999"
    target_daily_bars: int = TARGET_DAILY_BARS
    ewma_span: int = EWMA_SPAN
    fracdiff_threshold: float = FRACDIFF_THRES
    fracdiff_d: float = 0.0
    cusum_window: int = CUSUM_WINDOW
    cusum_multiplier: float = CUSUM_MULTIPLIER
    primary_span: int = int(MA_PRIMARY_MODEL.get("span", 20))
    model_path: str = "strategies/AL9999/output/models/meta_model.pkl"
    fixed_size: int = 1
    emit_orders: bool = False
    input_bar_mode: str = "minute"
    state_path: str = "strategies/AL9999/output/models/live_state.json"
    replay_tbm_path: str = ""
    replay_features_path: str = ""
    feature_nan_ratio_limit: float = 0.5
    feature_config: dict[str, Any] = field(default_factory=lambda: FEATURE_CONFIG.copy())
    tbm_config: dict[str, Any] = field(default_factory=lambda: TBM_CONFIG.copy())
    symbol_mapping: dict[str, str] = field(default_factory=dict)
    manual_symbol_overrides: dict[str, str] = field(default_factory=dict)
    meta_probability_threshold: float = float(META_MODEL_CONFIG.get("precision_threshold", 0.5))

    @classmethod
    def from_setting(cls, setting: dict[str, Any] | None = None) -> "Al9999LiveConfig":
        """
        Build config from a vn.py-style setting dictionary.
        """
        data = dict(setting or {})
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    def resolve_model_path(self, base_dir: str | Path | None = None) -> Path:
        """
        Resolve the model path relative to the repository when needed.
        """
        path = Path(self.model_path)
        if path.is_absolute():
            return path
        if base_dir is None:
            base_dir = Path(__file__).resolve().parents[2]
        return Path(base_dir) / path

    def resolve_state_path(self, base_dir: str | Path | None = None) -> Path:
        """
        Resolve the runtime state path relative to the repository when needed.
        """
        path = Path(self.state_path)
        if path.is_absolute():
            return path
        if base_dir is None:
            base_dir = Path(__file__).resolve().parents[2]
        return Path(base_dir) / path
