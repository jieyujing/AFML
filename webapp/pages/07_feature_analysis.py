"""特征分析页面 - 特征重要性和聚类分析"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from session import SessionManager
from components.sidebar import render_sidebar, navigate_to
from components.charts import render_heatmap, render_bar_chart, render_clustered_mda_chart

st.set_page_config(page_title="特征分析", page_icon="📊", layout="wide")

# ── Helper Functions ──────────────────────────────────────────────────


def _render_clustered_mda_section(
    features_data: pd.DataFrame,
    labels_data: pd.DataFrame,
) -> None:
    """渲染 Clustered MDA 分析部分"""
    from afmlkit.importance.clustering import cluster_features
    from afmlkit.importance.mda import clustered_mda

    st.markdown("#### Clustered MDA (聚类 MDA) 特征重要性")

    st.markdown("""
    **Clustered MDA** 通过打乱整个特征簇来测量重要性，消除替代效应：
    - 使用 ONC 算法自动确定最优聚类数
    - 同时打乱同一簇内的所有特征
    - 使用 Purged CV 和 Log-loss 计算样本外重要性
    """)

    # 参数配置
    with st.expander("Clustered MDA 参数", expanded=False):
        n_repeats = st.number_input("重复次数", min_value=1, max_value=50, value=10)
        n_splits = st.number_input("CV 折数", min_value=2, max_value=10, value=5)
        embargo_pct = st.number_input("Embargo 比例", min_value=0.0, max_value=0.1, value=0.01, step=0.005)
        random_state = st.number_input("随机种子", value=42)

    if not st.button("计算 Clustered MDA"):
        st.stop()

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # 准备数据
        status_text.text("准备数据...")
        numeric_cols = features_data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in ['open', 'high', 'low', 'close', 'volume']]

        X = features_data[feature_cols].dropna()

        # 获取 Triple Barrier 标签
        if isinstance(labels_data, pd.Series):
            y = labels_data
            t1 = y.index + pd.Timedelta(minutes=30)  # 默认 30 分钟
        else:
            y = labels_data.get('bin', labels_data.iloc[:, 0])
            t1 = labels_data.get('t1', y.index + pd.Timedelta(minutes=30))

        # 对齐索引
        common_idx = X.index.intersection(y.index).intersection(t1.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        t1 = t1.loc[common_idx]

        progress_bar.progress(20)

        # Step 1: 特征聚类
        status_text.text("执行特征聚类...")
        clusters = cluster_features(X)
        progress_bar.progress(40)

        # 显示聚类结果
        with st.expander("📊 聚类结果预览", expanded=True):
            st.write(f"**最优聚类数：** {len(clusters)} (通过 Silhouette Score 自动选择)")
            for cid, feats in sorted(clusters.items()):
                st.write(f"- **Cluster {cid}**: {', '.join(feats)}")

        # Step 2: 计算 Clustered MDA
        status_text.text("计算 Clustered MDA (可能需要几分钟)...")
        mda_df = clustered_mda(
            X=X,
            y=y,
            clusters=clusters,
            t1=t1,
            n_splits=n_splits,
            embargo_pct=embargo_pct,
            n_repeats=n_repeats,
            random_state=random_state
        )
        progress_bar.progress(80)

        # Step 3: 可视化
        status_text.text("渲染图表...")
        fig = render_clustered_mda_chart(mda_df, highlight_poison=True)
        st.plotly_chart(fig, use_container_width=True)

        # 毒药簇警告
        poison_clusters = mda_df[mda_df['mean_importance'] <= 0]
        if len(poison_clusters) > 0:
            st.warning("""
            ⚠️ **毒药簇警告：**

            以下簇在打乱后模型表现反而变好（重要性 ≤ 0），说明这些特征严重误导模型：
            """)
            for _, row in poison_clusters.iterrows():
                st.write(f"- **Cluster {row['cluster_id']}** (重要性={row['mean_importance']:.4f}): {', '.join(row['features'])}")
            st.write("""
            **建议：** 在实盘代码中直接 drop 掉这些特征！
            """)

        # 保存结果
        SessionManager.update('mda_results', mda_df)
        SessionManager.update('feature_clusters', clusters)

        status_text.text("分析完成!")
        progress_bar.progress(100)
        st.success("✅ Clustered MDA 分析完成")

    except Exception as e:
        st.error(f"Clustered MDA 分析失败：{str(e)}")
        import traceback
        st.code(traceback.format_exc())


def _render_simple_mda_section(
    features_data: pd.DataFrame,
    labels_data: pd.DataFrame,
) -> None:
    """渲染简单 MDA 分析部分（保留原有逻辑）"""
    st.markdown("#### 简单 MDA (单个特征) 特征重要性")

    st.markdown("""
    **MDA (Mean Decrease Accuracy)** 通过打乱特征值来测量特征重要性：
    - 打乱某个特征的值
    - 测量模型性能下降程度
    - 下降越多，特征越重要
    """)

    if not st.button("计算简单 MDA"):
        st.stop()

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # 准备数据
        status_text.text("准备数据...")
        numeric_cols = features_data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in ['open', 'high', 'low', 'close', 'volume']]

        X = features_data[feature_cols].dropna()

        # 获取标签
        if isinstance(labels_data, pd.Series):
            y = labels_data
        else:
            y = labels_data.get('bin', labels_data.iloc[:, 0])

        # 对齐索引
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        progress_bar.progress(20)

        # 使用随机森林计算重要性
        status_text.text("训练随机森林模型...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

        # 交叉验证
        cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        st.info(f"交叉验证准确率：{cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # 训练完整模型
        clf.fit(X, y)
        progress_bar.progress(50)

        # 获取特征重要性
        status_text.text("计算特征重要性...")
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)

        progress_bar.progress(70)

        # MDA 计算
        status_text.text("执行 MDA 分析...")
        from sklearn.metrics import accuracy_score

        baseline_score = accuracy_score(y, clf.predict(X))

        mda_scores = []
        for feat in feature_cols:
            X_permuted = X.copy()
            X_permuted[feat] = np.random.permutation(X_permuted[feat])
            permuted_score = accuracy_score(y, clf.predict(X_permuted))
            mda_scores.append(baseline_score - permuted_score)

        importance_df['mda_score'] = mda_scores
        importance_df = importance_df.sort_values('mda_score', ascending=False)

        progress_bar.progress(90)

        # 保存结果
        SessionManager.update('feature_importance', importance_df)
        SessionManager.update('model', clf)

        status_text.text("分析完成!")
        progress_bar.progress(100)

        st.success("✅ 特征重要性分析完成")

        # 显示重要性图
        st.markdown("#### 特征重要性 (MDA)")

        top_n = st.slider("显示 Top N 特征", 5, 50, 20)

        fig = render_bar_chart(
            importance_df.head(top_n)['mda_score'],
            title=f"Top {top_n} 特征重要性 (MDA)",
            orientation='h'
        )
        st.plotly_chart(fig, use_container_width=True)

        # 重要性表格
        st.markdown("#### 重要性排名")
        st.dataframe(importance_df.head(top_n))

    except Exception as e:
        st.error(f"特征重要性分析失败：{str(e)}")
        import traceback
        st.code(traceback.format_exc())


# ── Main Logic ────────────────────────────────────────────────────────

# 初始化会话
SessionManager.init_session()

# 渲染侧边栏
selected_page = render_sidebar()

# 处理导航
if selected_page != st.session_state.get('current_page', '首页'):
    navigate_to(selected_page)
    st.rerun()

st.title("📊 特征分析")

# 检查是否有特征数据
features = SessionManager.get('features')
labels = SessionManager.get('labels')

if features is None:
    st.warning("⚠️ 请先进行特征工程")
    st.stop()

st.info(f"当前特征数据：{len(features)} 行 × {len(features.columns)} 列")

# 步骤选择
step = st.radio(
    "选择步骤",
    ["1. 特征聚类", "2. 特征距离", "3. 特征重要性", "4. 特征选择"],
    horizontal=True
)

if step == "1. 特征聚类":
    st.markdown("### 1️⃣ 特征聚类分析")

    st.markdown("""
    **特征聚类** 用于识别高度相关的特征组，减少冗余：
    - 计算特征间的相关性矩阵
    - 使用层次聚类生成树状图
    - 基于聚类结果选择代表性特征
    """)

    if st.button("计算特征聚类"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # 选择数值特征
            status_text.text("准备特征数据...")
            numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c not in ['open', 'high', 'low', 'close', 'volume']]

            if len(feature_cols) < 2:
                st.error("特征数量不足")
                st.stop()

            feature_data = features[feature_cols].dropna()
            progress_bar.progress(20)

            # 计算相关性矩阵
            status_text.text("计算相关性矩阵...")
            corr_matrix = feature_data.corr()
            progress_bar.progress(40)

            # 转换为距离矩阵
            status_text.text("计算距离矩阵...")
            distance_matrix = np.sqrt((1 - corr_matrix) / 2)
            progress_bar.progress(60)

            # 层次聚类
            status_text.text("执行层次聚类...")
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import squareform

            linkage_matrix = linkage(squareform(distance_matrix.values), method='ward')
            progress_bar.progress(80)

            # 保存结果
            SessionManager.update('corr_matrix', corr_matrix)
            SessionManager.update('distance_matrix', distance_matrix)
            SessionManager.update('linkage_matrix', linkage_matrix)
            SessionManager.update('feature_cols', feature_cols)

            status_text.text("聚类完成!")
            progress_bar.progress(100)

            st.success("✅ 特征聚类完成")

            # 显示树状图
            st.markdown("#### 聚类树状图")

            import plotly.graph_objects as go

            fig = go.Figure()

            # 使用 plotly 绘制树状图
            fig.add_trace(go.Scatter(
                x=[0], y=[0],
                mode='markers',
                marker=dict(size=0),
                showlegend=False
            ))

            fig.update_layout(
                title="特征聚类树状图",
                xaxis=dict(title="特征"),
                yaxis=dict(title="距离"),
                template="plotly_white",
                height=600
            )

            # 用 matplotlib  dendrogram 数据创建 plotly 图形
            import matplotlib.pyplot as plt

            fig_mpl, ax = plt.subplots(figsize=(14, 8))
            dendrogram(linkage_matrix, labels=feature_cols, ax=ax)
            plt.xticks(rotation=90)
            plt.tight_layout()

            st.pyplot(fig_mpl)
            plt.close()

        except Exception as e:
            st.error(f"聚类分析失败：{str(e)}")
            import traceback
            st.code(traceback.format_exc())

elif step == "2. 特征距离":
    st.markdown("### 2️⃣ 特征距离热力图")

    distance_matrix = SessionManager.get('distance_matrix')
    corr_matrix = SessionManager.get('corr_matrix')

    if distance_matrix is None:
        st.warning("请先执行特征聚类")
        if st.button("执行聚类分析"):
            navigate_to("4️⃣ 特征分析")
            st.rerun()
    else:
        st.markdown("#### 特征距离矩阵")

        # 显示热力图
        fig = render_heatmap(
            distance_matrix.round(3),
            title="特征距离矩阵",
            colorscale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 特征相关性矩阵")

        fig_corr = render_heatmap(
            corr_matrix.round(3),
            title="特征相关性矩阵",
            colorscale="RdBu"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # 高相关性特征对
        st.markdown("#### 高相关性特征对 (>0.8)")

        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append({
                        '特征 1': corr_matrix.columns[i],
                        '特征 2': corr_matrix.columns[j],
                        '相关系数': corr_val
                    })

        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs)
            high_corr_df = high_corr_df.sort_values('相关系数', ascending=False)
            st.dataframe(high_corr_df.head(20))
        else:
            st.info("没有发现高相关性特征对")

if step == "3. 特征重要性":
    st.markdown("### 3️⃣ 特征重要性分析")

    features_data = SessionManager.get('features')
    labels_data = SessionManager.get('labels')

    if features_data is None or labels_data is None:
        st.warning("请确保已有特征和标签数据")
    else:
        # 模式选择
        mode = st.radio(
            "分析模式",
            ["快速 MDA (单个特征)", "Clustered MDA (推荐)"],
            horizontal=True
        )

        if mode == "Clustered MDA (推荐)":
            _render_clustered_mda_section(features_data, labels_data)
        else:
            _render_simple_mda_section(features_data, labels_data)

elif step == "4. 特征选择":
    st.markdown("### 4️⃣ 特征选择")

    feature_importance = SessionManager.get('feature_importance')

    if feature_importance is None:
        st.warning("请先计算特征重要性")
    else:
        st.markdown("#### 特征选择方法")

        method = st.selectbox(
            "选择方法",
            ["importance_threshold", "top_k", "cumulative_importance"]
        )

        if method == "importance_threshold":
            threshold = st.slider("重要性阈值", 0.0, 0.1, 0.01, 0.001)
            selected = feature_importance[feature_importance['mda_score'] > threshold]
        elif method == "top_k":
            k = st.number_input("选择 Top K 个特征", min_value=1, max_value=len(feature_importance), value=10)
            selected = feature_importance.head(k)
        else:
            cumulative = st.slider("累积重要性阈值", 0.5, 1.0, 0.9, 0.05)
            feature_importance['cumulative'] = feature_importance['mda_score'].cumsum() / feature_importance['mda_score'].sum()
            selected = feature_importance[feature_importance['cumulative'] <= cumulative]

        st.markdown("#### 选中的特征")
        st.write(selected['feature'].tolist())

        # 保存选择
        if st.button("保存特征选择"):
            SessionManager.update('selected_features', selected['feature'].tolist())
            st.success("特征选择已保存")

# 导航按钮
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("⬅️ 返回标签生成", use_container_width=True):
        navigate_to("3️⃣ 标签生成")
        st.rerun()

with col3:
    if SessionManager.get('feature_importance') is not None:
        if st.button("前往模型训练 ➡️", use_container_width=True):
            navigate_to("5️⃣ 模型训练")
            st.rerun()
