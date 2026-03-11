"""Clustered MDA 集成测试"""
import numpy as np
import pandas as pd
import pytest
from webapp.components.charts import render_clustered_mda_chart
from afmlkit.importance.clustering import cluster_features
from afmlkit.importance.mda import clustered_mda


@pytest.fixture
def sample_features_and_labels():
    """生成模拟特征和 Triple Barrier 标签"""
    np.random.seed(42)
    n = 500

    dates = pd.date_range('2023-01-01', periods=n, freq='5min')

    # 生成特征
    data = {
        f'feat_{i}': np.random.randn(n) * 0.01
        for i in range(10)
    }
    # 添加一些相关特征
    data['feat_correlated'] = data['feat_0'] * 0.8 + np.random.randn(n) * 0.002

    X = pd.DataFrame(data, index=dates)

    # 生成 Triple Barrier 标签
    y = pd.Series(
        np.random.choice([-1, 0, 1], n),
        index=dates,
        name='bin'
    )
    t1 = pd.Series(dates + pd.Timedelta(minutes=30), index=dates, name='t1')

    return X, y, t1


def test_clustered_mda_full_pipeline(sample_features_and_labels):
    """测试完整 Clustered MDA 流程"""
    X, y, t1 = sample_features_and_labels

    # Step 1: 特征聚类
    clusters = cluster_features(X)
    assert len(clusters) >= 1
    assert len(clusters) <= len(X.columns)

    # Step 2: Clustered MDA
    mda_df = clustered_mda(
        X=X,
        y=y,
        clusters=clusters,
        t1=t1,
        n_splits=3,  # 减少折数加快测试
        n_repeats=2,
        random_state=42
    )

    # 验证结果
    assert len(mda_df) == len(clusters)
    assert 'cluster_id' in mda_df.columns
    assert 'mean_importance' in mda_df.columns
    assert 'std_importance' in mda_df.columns

    # Step 3: 可视化
    fig = render_clustered_mda_chart(mda_df)
    assert fig is not None
    assert len(fig.data) >= 1


def test_clustered_mda_with_poison_clusters(sample_features_and_labels):
    """测试毒药簇检测"""
    X, y, t1 = sample_features_and_labels

    clusters = cluster_features(X)
    mda_df = clustered_mda(
        X=X, y=y, clusters=clusters, t1=t1,
        n_splits=3, n_repeats=1, random_state=42
    )

    # 验证图表能正确处理毒药簇
    fig = render_clustered_mda_chart(mda_df, highlight_poison=True)
    assert fig is not None


def test_clustered_mda_missing_labels():
    """测试缺失标签时的降级处理"""
    X = pd.DataFrame({
        'feat_1': np.random.randn(100),
        'feat_2': np.random.randn(100)
    })

    # 没有提供 y 和 t1，应该抛出有意义的错误
    with pytest.raises((ValueError, TypeError)):
        clustered_mda(X=X, y=None, clusters={1: ['feat_1']}, t1=None)
