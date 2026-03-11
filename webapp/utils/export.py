"""导出工具"""
import pandas as pd
from pathlib import Path
from typing import Union, Optional, Dict, Any
import io


def export_to_csv(
    df: pd.DataFrame,
    filepath: Optional[Union[str, Path]] = None
) -> Optional[str]:
    """导出为 CSV

    Args:
        df: DataFrame
        filepath: 保存路径（可选）

    Returns:
        如果未指定 filepath，返回 CSV 字符串
    """
    if filepath:
        df.to_csv(filepath, index=True)
    else:
        return df.to_csv(index=True)


def export_to_parquet(
    df: pd.DataFrame,
    filepath: Union[str, Path]
):
    """导出为 Parquet

    Args:
        df: DataFrame
        filepath: 保存路径
    """
    df.to_parquet(filepath, index=True)


def export_to_excel(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    sheet_name: str = 'Sheet1'
):
    """导出为 Excel

    Args:
        df: DataFrame
        filepath: 保存路径
        sheet_name: 工作表名称
    """
    df.to_excel(filepath, sheet_name=sheet_name, index=True)


def create_report(
    data: Dict[str, Any],
    title: str = "分析报告"
) -> str:
    """创建 HTML 报告

    Args:
        data: 报告数据
        title: 报告标题

    Returns:
        HTML 字符串
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ display: inline-block; margin: 10px; padding: 15px;
                     background: #f9f9f9; border-radius: 5px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
            .metric-label {{ font-size: 14px; color: #666; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
    """

    # 添加指标
    if 'metrics' in data:
        html += "<div>"
        for name, value in data['metrics'].items():
            html += f"""
            <div class="metric">
                <div class="metric-value">{value:.4f if isinstance(value, float) else value}</div>
                <div class="metric-label">{name}</div>
            </div>
            """
        html += "</div>"

    # 添加表格
    for table_name, df in data.items():
        if isinstance(df, pd.DataFrame):
            html += f"<h2>{table_name}</h2>"
            html += df.head(100).to_html()

    html += """
    </body>
    </html>
    """

    return html


def save_report(
    content: str,
    filepath: Union[str, Path],
    fmt: str = "html"
):
    """保存报告

    Args:
        content: 报告内容
        filepath: 保存路径
        fmt: 文件格式
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


def download_button(
    data: Union[pd.DataFrame, str, bytes],
    filename: str,
    label: str = "下载",
    mime: str = "text/csv"
) -> None:
    """Streamlit 下载按钮

    Args:
        data: 数据
        filename: 文件名
        label: 按钮标签
        mime: MIME 类型
    """
    import streamlit as st

    if isinstance(data, pd.DataFrame):
        data = data.to_csv(index=True)

    if isinstance(data, str):
        data = data.encode('utf-8')

    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime
    )
