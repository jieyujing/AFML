"""Tests for charts.py components"""
import pandas as pd
import pytest
from webapp.components.charts import render_clustered_mda_chart


def test_render_clustered_mda_chart_basic():
    """测试基本图表渲染"""
    mda_df = pd.DataFrame({
        'cluster_id': [1, 2, 3],
        'features': [['feat_a'], ['feat_b'], ['feat_c']],
        'mean_importance': [0.05, 0.03, 0.01],
        'std_importance': [0.01, 0.005, 0.002]
    })

    fig = render_clustered_mda_chart(mda_df)

    assert fig is not None
    # 误差线是 Bar 的一部分，不是单独的 trace
    assert len(fig.data) == 1
    # 验证误差线存在
    assert fig.data[0].error_x is not None
    assert fig.data[0].error_x.color == 'red'


def test_render_clustered_mda_chart_poison_highlight():
    """测试毒药簇高亮（重要性 <= 0）"""
    mda_df = pd.DataFrame({
        'cluster_id': [1, 2, 3],
        'features': [['feat_a'], ['feat_b'], ['feat_c']],
        'mean_importance': [0.05, -0.02, 0.01],  # Cluster 2 is poison
        'std_importance': [0.01, 0.005, 0.002]
    })

    fig = render_clustered_mda_chart(mda_df, highlight_poison=True)

    # 验证毒药簇使用红色
    assert fig is not None
    # 验证有一个条形是红色（毒药）
    bar_trace = fig.data[0]
    # 排序后，最重要的（0.05）在第一行，毒药（-0.02）在最后一行
    assert '#d62728' in bar_trace.marker.color  # 至少有一个红色


def test_render_clustered_mda_chart_empty():
    """测试空 DataFrame 处理"""
    mda_df = pd.DataFrame()

    with pytest.raises(ValueError, match="MDA DataFrame cannot be empty"):
        render_clustered_mda_chart(mda_df)


def test_render_clustered_mda_chart_sorting():
    """测试按重要性降序排序"""
    mda_df = pd.DataFrame({
        'cluster_id': [1, 2, 3],
        'features': [['feat_a'], ['feat_b'], ['feat_c']],
        'mean_importance': [0.01, 0.05, 0.03],  # 未排序
        'std_importance': [0.002, 0.01, 0.005]
    })

    fig = render_clustered_mda_chart(mda_df)

    # 验证第一个条形是最重要（0.05）
    # y 轴标签应该是 Cluster 2
    assert 'Cluster 2' in fig.data[0].y[0]


def test_render_clustered_mda_chart_poison_color_position():
    """测试毒药簇在正确位置显示红色"""
    mda_df = pd.DataFrame({
        'cluster_id': [1, 2, 3],
        'features': [['feat_a'], ['feat_b'], ['feat_c']],
        'mean_importance': [-0.02, 0.05, 0.03],  # Cluster 1 is poison
        'std_importance': [0.01, 0.005, 0.002]
    })

    fig = render_clustered_mda_chart(mda_df, highlight_poison=True)

    bar_trace = fig.data[0]
    # 排序后：0.05 ( Cluster 2), 0.03 (Cluster 3), -0.02 (Cluster 1)
    # 最后一个应该是红色
    assert bar_trace.marker.color[2] == '#d62728'  # 最后一个是红色
