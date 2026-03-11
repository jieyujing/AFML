"""配置管理模块 - 使用 YAML/JSON 保存用户配置"""
import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


class ConfigManager:
    """配置管理器 - 支持 YAML 和 JSON 格式"""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.experiments_dir = self.base_dir / "experiments"
        self._ensure_dirs()

    def _ensure_dirs(self):
        """确保必要目录存在"""
        self.experiments_dir.mkdir(exist_ok=True)

    def save_config(self, config: Dict[str, Any], name: str, fmt: str = "yaml") -> Path:
        """保存配置到文件

        Args:
            config: 配置字典
            name: 配置名称
            fmt: 文件格式 ('yaml' 或 'json')

        Returns:
            保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}"

        if fmt == "yaml":
            filepath = self.experiments_dir / f"{filename}.yaml"
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        else:
            filepath = self.experiments_dir / f"{filename}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

        return filepath

    def load_config(self, filepath: str) -> Dict[str, Any]:
        """从文件加载配置

        Args:
            filepath: 配置文件路径

        Returns:
            配置字典
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"配置文件不存在：{filepath}")

        suffix = filepath.suffix.lower()
        with open(filepath, 'r', encoding='utf-8') as f:
            if suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif suffix == '.json':
                return json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式：{suffix}")

    def list_experiments(self) -> list:
        """列出所有实验配置

        Returns:
            实验文件列表
        """
        experiments = []
        for pattern in ['*.yaml', '*.yml', '*.json']:
            experiments.extend(self.experiments_dir.glob(pattern))
        return sorted(experiments, key=lambda x: x.stat().st_mtime, reverse=True)

    def delete_experiment(self, name: str) -> bool:
        """删除实验配置

        Args:
            name: 实验名称

        Returns:
            是否删除成功
        """
        filepath = self.experiments_dir / name
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    def get_default_config(self, section: str) -> Dict[str, Any]:
        """获取默认配置

        Args:
            section: 配置部分 (e.g., 'bar', 'feature', 'label', 'model', 'backtest')

        Returns:
            默认配置字典
        """
        defaults = {
            'bar': {
                'type': 'time',  # time, tick, volume, dollar, cusum, imbalance
                'interval': '1min',
                'threshold': 1000,  # for volume/dollar bars
                'cusum_threshold': 0.05,
            },
            'feature': {
                'transforms': [],
                'window_sizes': [10, 20, 50],
                'features': ['returns', 'volatility', 'momentum']
            },
            'label': {
                'method': 'tbm',  # triple barrier method
                'time_horizon': 10,
                'profit_barrier': 0.02,
                'loss_barrier': 0.02,
                'sample_weight_method': 'concurrency'
            },
            'model': {
                'type': 'random_forest',
                'n_estimators': 100,
                'max_depth': 10,
                'cv_method': 'purged',
                'n_splits': 5
            },
            'backtest': {
                'initial_capital': 1000000,
                'commission': 0.001,
                'slippage': 0.0005,
            }
        }
        return defaults.get(section, {})
