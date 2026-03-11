"""会话状态管理 - Streamlit session_state 封装"""
import streamlit as st
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd


@dataclass
class SessionData:
    """会话数据结构"""
    # 数据相关
    raw_data: Optional[pd.DataFrame] = None
    bar_data: Optional[pd.DataFrame] = None
    features: Optional[pd.DataFrame] = None
    labels: Optional[pd.Series] = None
    sample_weights: Optional[pd.Series] = None

    # 配置相关
    bar_config: Dict[str, Any] = field(default_factory=dict)
    feature_config: Dict[str, Any] = field(default_factory=dict)
    label_config: Dict[str, Any] = field(default_factory=dict)
    model_config: Dict[str, Any] = field(default_factory=dict)
    backtest_config: Dict[str, Any] = field(default_factory=dict)

    # 模型相关
    model: Any = None
    model_results: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Optional[pd.DataFrame] = None

    # 回测相关
    backtest_results: Dict[str, Any] = field(default_factory=dict)

    # 可视化相关
    plots: Dict[str, Any] = field(default_factory=dict)

    # 状态相关
    current_step: int = 0
    is_processing: bool = False
    last_updated: Optional[datetime] = None

    # 实验相关
    experiment_name: str = ""
    experiment_notes: str = ""


class SessionManager:
    """会话状态管理器"""

    KEYS = [
        'raw_data', 'bar_data', 'dollar_bars', 'features', 'labels', 'sample_weights',
        'bar_config', 'feature_config', 'label_config', 'model_config', 'backtest_config',
        'model', 'model_results', 'feature_importance', 'feature_metadata',
        'backtest_results', 'plots',
        'current_step', 'is_processing', 'last_updated',
        'experiment_name', 'experiment_notes',
        'iid_results', 'iid_score_df', 'best_freq', 'generation_time'
    ]

    @staticmethod
    def init_session():
        """初始化会话状态"""
        for key in SessionManager.KEYS:
            if key not in st.session_state:
                if key == 'plots':
                    st.session_state[key] = {}
                elif key in ['bar_config', 'feature_config', 'label_config',
                            'model_config', 'backtest_config', 'model_results',
                            'backtest_results', 'feature_metadata']:
                    st.session_state[key] = {}
                elif key == 'current_step':
                    st.session_state[key] = 0
                elif key == 'is_processing':
                    st.session_state[key] = False
                else:
                    st.session_state[key] = None

        if 'last_updated' not in st.session_state or st.session_state.last_updated is None:
            st.session_state.last_updated = datetime.now()

    @staticmethod
    def update(key: str, value: Any):
        """更新会话状态

        Args:
            key: 状态键
            value: 状态值
        """
        st.session_state[key] = value
        st.session_state.last_updated = datetime.now()

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """获取会话状态

        Args:
            key: 状态键
            default: 默认值（如果 key 不存在）

        Returns:
            状态值
        """
        return st.session_state.get(key, default)

    @staticmethod
    def get_all() -> Dict[str, Any]:
        """获取所有会话状态

        Returns:
            所有状态字典
        """
        return {key: st.session_state.get(key) for key in SessionManager.KEYS}

    @staticmethod
    def reset_data():
        """重置数据相关状态"""
        data_keys = ['raw_data', 'bar_data', 'features', 'labels', 'sample_weights']
        for key in data_keys:
            st.session_state[key] = None

    @staticmethod
    def reset_all():
        """重置所有会话状态"""
        for key in SessionManager.KEYS:
            if key == 'plots':
                st.session_state[key] = {}
            elif key in ['bar_config', 'feature_config', 'label_config',
                        'model_config', 'backtest_config', 'model_results',
                        'backtest_results', 'feature_metadata']:
                st.session_state[key] = {}
            elif key == 'current_step':
                st.session_state[key] = 0
            elif key == 'is_processing':
                st.session_state[key] = False
            elif key == 'last_updated':
                st.session_state[key] = datetime.now()
            else:
                st.session_state[key] = None

    @staticmethod
    def next_step():
        """进入下一步"""
        st.session_state.current_step += 1

    @staticmethod
    def prev_step():
        """返回上一步"""
        if st.session_state.current_step > 0:
            st.session_state.current_step -= 1

    @staticmethod
    def set_processing(status: bool):
        """设置处理状态"""
        st.session_state.is_processing = status

    @staticmethod
    def is_processing() -> bool:
        """检查是否在_processing"""
        return st.session_state.get('is_processing', False)

    @staticmethod
    def save_snapshot(name: str = ""):
        """保存当前会话快照（用于实验）"""
        if not name:
            name = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        snapshot = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'data': SessionManager.get_all()
        }

        # 保存到 session_state 的快照列表
        if 'snapshots' not in st.session_state:
            st.session_state.snapshots = []
        st.session_state.snapshots.append(snapshot)

        return name

    @staticmethod
    def load_snapshot(name: str) -> bool:
        """加载会话快照

        Args:
            name: 快照名称

        Returns:
            是否加载成功
        """
        if 'snapshots' not in st.session_state:
            return False

        for snapshot in st.session_state.snapshots:
            if snapshot['name'] == name:
                for key, value in snapshot['data'].items():
                    st.session_state[key] = value
                return True
        return False

    @staticmethod
    def list_snapshots() -> List[Dict[str, str]]:
        """列出所有快照

        Returns:
            快照列表
        """
        if 'snapshots' not in st.session_state:
            return []
        return [
            {'name': s['name'], 'timestamp': s['timestamp']}
            for s in st.session_state.snapshots
        ]
