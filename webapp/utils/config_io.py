"""配置 IO 工具"""
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Union
from datetime import datetime


def save_yaml(data: Dict[str, Any], filepath: Union[str, Path]):
    """保存为 YAML 文件

    Args:
        data: 数据字典
        filepath: 保存路径
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def load_yaml(filepath: Union[str, Path]) -> Dict[str, Any]:
    """加载 YAML 文件

    Args:
        filepath: 文件路径

    Returns:
        数据字典
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_json(data: Dict[str, Any], filepath: Union[str, Path], indent: int = 2):
    """保存为 JSON 文件

    Args:
        data: 数据字典
        filepath: 保存路径
        indent: 缩进空格数
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """加载 JSON 文件

    Args:
        filepath: 文件路径

    Returns:
        数据字典
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_experiment(
    experiment_data: Dict[str, Any],
    name: str,
    base_dir: Union[str, Path]
) -> Path:
    """保存实验数据

    Args:
        experiment_data: 实验数据字典
        name: 实验名称
        base_dir: 基础目录

    Returns:
        保存的文件路径
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = base_dir / f"{name}_{timestamp}.yaml"

    save_yaml(experiment_data, filepath)

    return filepath


def load_experiment(filepath: Union[str, Path]) -> Dict[str, Any]:
    """加载实验数据

    Args:
        filepath: 文件路径

    Returns:
        实验数据字典
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix in ['.yaml', '.yml']:
        return load_yaml(filepath)
    elif suffix == '.json':
        return load_json(filepath)
    else:
        raise ValueError(f"不支持的配置文件格式：{suffix}")


def list_experiments(base_dir: Union[str, Path]) -> list:
    """列出所有实验

    Args:
        base_dir: 基础目录

    Returns:
        实验文件列表
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []

    experiments = []
    for pattern in ['*.yaml', '*.yml', '*.json']:
        experiments.extend(base_dir.glob(pattern))

    return sorted(experiments, key=lambda x: x.stat().st_mtime, reverse=True)
