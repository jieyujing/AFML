"""参数编辑器组件"""
import streamlit as st
from typing import Any, Dict, Callable


def render_param_editor(
    title: str,
    params: Dict[str, Any],
    key_prefix: str = "",
    on_change: Callable = None
) -> Dict[str, Any]:
    """渲染参数编辑器

    Args:
        title: 编辑器标题
        params: 参数字典
        key_prefix: 键前缀
        on_change: 变更回调

    Returns:
        更新后的参数字典
    """
    with st.expander(title, expanded=True):
        updated_params = params.copy()

        for param_name, param_value in params.items():
            key = f"{key_prefix}_{param_name}"

            if isinstance(param_value, bool):
                updated_params[param_name] = st.checkbox(
                    param_name,
                    value=param_value,
                    key=key
                )
            elif isinstance(param_value, int):
                updated_params[param_name] = st.number_input(
                    param_name,
                    value=param_value,
                    step=1,
                    key=key
                )
            elif isinstance(param_value, float):
                updated_params[param_name] = st.number_input(
                    param_name,
                    value=param_value,
                    step=0.01,
                    key=key
                )
            elif isinstance(param_value, str):
                updated_params[param_name] = st.text_input(
                    param_name,
                    value=param_value,
                    key=key
                )
            elif isinstance(param_value, list):
                updated_params[param_name] = st.multiselect(
                    param_name,
                    options=param_value,
                    default=param_value,
                    key=key
                )
            else:
                st.text(f"{param_name}: {param_value}")

        if on_change:
            on_change(updated_params)

        return updated_params


def render_dict_editor(
    title: str,
    data: Dict[str, Any],
    key: str = ""
) -> Dict[str, Any]:
    """渲染字典编辑器（用于复杂配置）

    Args:
        title: 编辑器标题
        data: 要编辑的字典
        key: Streamlit 键

    Returns:
        编辑后的字典
    """
    with st.expander(title, expanded=True):
        edited_data = data.copy()

        for k, v in data.items():
            col1, col2 = st.columns([1, 2])

            with col1:
                st.write(f"**{k}**")

            with col2:
                if isinstance(v, bool):
                    edited_data[k] = st.checkbox(
                        "值",
                        value=v,
                        key=f"{key}_{k}"
                    )
                elif isinstance(v, (int, float)):
                    edited_data[k] = st.number_input(
                        "值",
                        value=v,
                        key=f"{key}_{k}"
                    )
                elif isinstance(v, str):
                    edited_data[k] = st.text_input(
                        "值",
                        value=v,
                        key=f"{key}_{k}"
                    )
                else:
                    st.write(str(v))

        return edited_data
