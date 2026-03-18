"""模型训练页面"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="模型训练", page_icon="🤖", layout="wide")

from session import SessionManager
from components.sidebar import render_sidebar, navigate_to
from components.charts import render_bar_chart, render_line_chart, display_metrics

# 初始化会话
SessionManager.init_session()

# 渲染侧边栏
selected_page = render_sidebar()

# 处理导航
if selected_page != st.session_state.get('current_page', '首页'):
    navigate_to(selected_page)
    st.rerun()

st.title("🤖 模型训练")

# 检查数据
features = SessionManager.get('features')
labels = SessionManager.get('labels')

if features is None:
    st.warning("⚠️ 请先进行特征工程")
    st.stop()

st.info(f"当前数据：{len(features)} 行")

# 步骤选择
step = st.radio(
    "选择步骤",
    ["1. 模型配置", "2. 交叉验证", "3. 模型训练", "4. 模型评估"],
    horizontal=True
)

if step == "1. 模型配置":
    st.markdown("### 1️⃣ 模型配置")

    st.markdown("#### 模型选择")

    model_type = st.selectbox(
        "选择模型类型",
        ["random_forest", "gradient_boosting", "xgboost"],
        format_func=lambda x: {
            'random_forest': '随机森林 (Random Forest)',
            'gradient_boosting': '梯度提升 (Gradient Boosting)',
            'xgboost': 'XGBoost'
        }.get(x, x)
    )

    st.markdown("#### 超参数配置")

    col1, col2, col3 = st.columns(3)

    with col1:
        n_estimators = st.number_input("树的数量", min_value=10, max_value=1000, value=100, step=10)
    with col2:
        max_depth = st.number_input("最大深度", min_value=2, max_value=50, value=10)
    with col3:
        min_samples = st.number_input("最小样本数", min_value=2, max_value=100, value=5)

    st.markdown("#### 交叉验证配置")

    cv_method = st.selectbox(
        "CV 方法",
        ["purged", "cpcv", "kfold"],
        format_func=lambda x: {
            'purged': 'Purged CV (去污交叉验证)',
            'cpcv': 'CPCV (组合式 Purged CV)',
            'kfold': '普通 K 折交叉验证'
        }.get(x, x)
    )

    n_splits = st.slider("折数", 3, 10, 5)

    # 保存配置
    if st.button("保存模型配置"):
        model_config = {
            'type': model_type,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples': min_samples,
            'cv_method': cv_method,
            'n_splits': n_splits
        }
        SessionManager.update('model_config', model_config)
        st.success("模型配置已保存")
        st.json(model_config)

elif step == "2. 交叉验证":
    st.markdown("### 2️⃣ 交叉验证配置")

    model_config = SessionManager.get('model_config', {})

    if not model_config:
        st.warning("请先配置模型参数")
    else:
        st.json(model_config)

        st.markdown("#### Purged CV 参数")

        samples_info = st.number_input("每个折的样本数", min_value=0, value=0, disabled=True)

        embargo = st.slider("Embargo 比例", 0.0, 0.2, 0.05, 0.01,
                           help="每个验证集后排除的样本比例，防止信息泄露")

        if st.button("执行交叉验证"):
            st.info("使用 Purged CV 进行交叉验证...")

            try:
                # 准备数据
                numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
                feature_cols = [c for c in numeric_cols if c not in ['open', 'high', 'low', 'close', 'volume']]

                X = features[feature_cols].dropna()

                if isinstance(labels, pd.Series):
                    y = labels
                else:
                    y = labels.get('bin', labels.iloc[:, 0] if hasattr(labels, 'iloc') else labels)

                common_idx = X.index.intersection(y.index)
                X = X.loc[common_idx]
                y = y.loc[common_idx]

                # 创建 Purged CV
                from afmlkit.validation import PurgedKFold

                cv = PurgedKFold(
                    n_splits=model_config.get('n_splits', 5),
                    samples_info=None,
                    embargo=embargo
                )

                # 执行交叉验证
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import accuracy_score, classification_report

                scores = []
                fold_results = []

                progress_bar = st.progress(0)

                for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    clf = RandomForestClassifier(
                        n_estimators=model_config.get('n_estimators', 100),
                        max_depth=model_config.get('max_depth', 10),
                        min_samples_leaf=model_config.get('min_samples', 5),
                        random_state=42,
                        n_jobs=-1
                    )

                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                    score = accuracy_score(y_test, y_pred)
                    scores.append(score)
                    fold_results.append({
                        'fold': i + 1,
                        'accuracy': score,
                        'train_size': len(X_train),
                        'test_size': len(X_test)
                    })

                    progress_bar.progress((i + 1) / model_config.get('n_splits', 5))

                # 显示结果
                results_df = pd.DataFrame(fold_results)
                st.markdown("#### 各折结果")
                st.dataframe(results_df)

                mean_score = np.mean(scores)
                std_score = np.std(scores)

                st.success(f"✅ 交叉验证完成")
                st.metric("平均准确率", f"{mean_score:.4f}")
                st.metric("标准差", f"{std_score:.4f}")

                # 保存结果
                SessionManager.update('cv_results', results_df)
                SessionManager.update('cv_scores', scores)

            except ImportError:
                st.warning("afmlkit.validation 模块不可用，使用普通 KFold")

                from sklearn.model_selection import KFold
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import accuracy_score

                kf = KFold(n_splits=model_config.get('n_splits', 5), shuffle=True, random_state=42)

                scores = []
                for train_idx, test_idx in kf.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    clf = RandomForestClassifier(
                        n_estimators=model_config.get('n_estimators', 100),
                        max_depth=model_config.get('max_depth', 10),
                        random_state=42
                    )
                    clf.fit(X_train, y_train)
                    scores.append(accuracy_score(y_test, clf.predict(X_test)))

                st.success(f"平均准确率：{np.mean(scores):.4f}")

            except Exception as e:
                st.error(f"交叉验证失败：{str(e)}")

elif step == "3. 模型训练":
    st.markdown("### 3️⃣ 模型训练")

    model_config = SessionManager.get('model_config', {})

    if st.button("开始训练模型"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # 准备数据
            status_text.text("准备数据...")
            numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c not in ['open', 'high', 'low', 'close', 'volume']]

            X = features[feature_cols].dropna()

            if isinstance(labels, pd.Series):
                y = labels
            else:
                y = labels.get('bin', labels.iloc[:, 0] if hasattr(labels, 'iloc') else labels)

            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]

            progress_bar.progress(20)

            # 选择模型
            status_text.text(f"训练 {model_config.get('type', 'random_forest')}...")

            model_type = model_config.get('type', 'random_forest')

            if model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                clf = RandomForestClassifier(
                    n_estimators=model_config.get('n_estimators', 100),
                    max_depth=model_config.get('max_depth', 10),
                    min_samples_leaf=model_config.get('min_samples', 5),
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == 'gradient_boosting':
                from sklearn.ensemble import GradientBoostingClassifier
                clf = GradientBoostingClassifier(
                    n_estimators=model_config.get('n_estimators', 100),
                    max_depth=model_config.get('max_depth', 10),
                    min_samples_leaf=model_config.get('min_samples', 5),
                    random_state=42
                )
            else:
                try:
                    import xgboost as xgb
                    clf = xgb.XGBClassifier(
                        n_estimators=model_config.get('n_estimators', 100),
                        max_depth=model_config.get('max_depth', 10),
                        min_child_weight=model_config.get('min_samples', 5),
                        random_state=42,
                        n_jobs=-1
                    )
                except ImportError:
                    st.warning("XGBoost 不可用，使用随机森林")
                    from sklearn.ensemble import RandomForestClassifier
                    clf = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42
                    )

            progress_bar.progress(40)

            # 训练
            status_text.text("正在训练...")
            clf.fit(X, y)
            progress_bar.progress(80)

            # 预测
            status_text.text("评估模型...")
            y_pred = clf.predict(X)
            y_proba = clf.predict_proba(X) if hasattr(clf, 'predict_proba') else None

            progress_bar.progress(90)

            # 保存模型
            SessionManager.update('model', clf)
            SessionManager.update('y_pred', y_pred)
            SessionManager.update('y_proba', y_proba)
            SessionManager.update('feature_names', feature_cols)

            status_text.text("训练完成!")
            progress_bar.progress(100)

            st.success(f"✅ 模型训练完成")

            # 显示训练准确率
            from sklearn.metrics import accuracy_score, classification_report
            train_acc = accuracy_score(y, y_pred)
            st.metric("训练准确率", f"{train_acc:.4f}")

        except Exception as e:
            st.error(f"模型训练失败：{str(e)}")
            import traceback
            st.code(traceback.format_exc())

elif step == "4. 模型评估":
    st.markdown("### 4️⃣ 模型评估")

    model = SessionManager.get('model')
    y_pred = SessionManager.get('y_pred')

    if model is None:
        st.warning("请先训练模型")
    else:
        # 准备真实标签
        if isinstance(labels, pd.Series):
            y_true = labels
        else:
            y_true = labels.get('bin', labels.iloc[:, 0] if hasattr(labels, 'iloc') else labels)

        if y_pred is None:
            st.warning("没有预测结果")
        else:
            # 对齐
            if hasattr(y_pred, '__len__') and len(y_pred) == len(y_true):
                # 计算指标
                from sklearn.metrics import (
                    accuracy_score, precision_score, recall_score,
                    f1_score, confusion_matrix, classification_report
                )

                st.markdown("#### 分类指标")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    acc = accuracy_score(y_true, y_pred)
                    st.metric("准确率", f"{acc:.4f}")

                with col2:
                    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                    st.metric("精确率", f"{prec:.4f}")

                with col3:
                    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                    st.metric("召回率", f"{rec:.4f}")

                with col4:
                    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                    st.metric("F1 分数", f"{f1:.4f}")

                # 混淆矩阵
                st.markdown("#### 混淆矩阵")

                cm = confusion_matrix(y_true, y_pred)
                cm_df = pd.DataFrame(cm)

                import plotly.graph_objects as go

                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['-1', '0', '1'] if cm.shape[1] == 3 else list(range(cm.shape[1])),
                    y=['-1', '0', '1'] if cm.shape[0] == 3 else list(range(cm.shape[0])),
                    colorscale='Blues'
                ))

                fig.update_layout(
                    title="混淆矩阵",
                    xaxis_title="预测",
                    yaxis_title="真实",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

                # 特征重要性
                st.markdown("#### 特征重要性")

                if hasattr(model, 'feature_importances_'):
                    feature_names = SessionManager.get('feature_names', [])

                    importance_df = pd.DataFrame({
                        'feature': feature_names[:len(model.feature_importances_)],
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)

                    top_n = st.slider("显示 Top N", 5, 30, 15)

                    fig = render_bar_chart(
                        importance_df.head(top_n)['importance'],
                        title="特征重要性",
                        orientation='h'
                    )
                    st.plotly_chart(fig, use_container_width=True)

# 导航按钮
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("⬅️ 返回特征分析", use_container_width=True):
        navigate_to("4️⃣ 特征分析")
        st.rerun()

with col3:
    if SessionManager.get('model') is not None:
        if st.button("前往回测评估 ➡️", use_container_width=True):
            navigate_to("6️⃣ 回测评估")
            st.rerun()
