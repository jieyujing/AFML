"""回测与评估页面"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="回测评估", page_icon="📈", layout="wide")

from session import SessionManager
from components.sidebar import render_sidebar, navigate_to
from components.charts import render_equity_curve, render_drawdown_chart, render_line_chart, display_metrics

# 初始化会话
SessionManager.init_session()

# 渲染侧边栏
selected_page = render_sidebar()

# 处理导航
if selected_page != st.session_state.get('current_page', '首页'):
    navigate_to(selected_page)
    st.rerun()

st.title("📈 回测与绩效评估")

# 检查数据
model = SessionManager.get('model')
bar_data = SessionManager.get('bar_data')

if bar_data is None:
    st.warning("⚠️ 请先导入数据并构建 K 线")
    st.stop()

st.info(f"当前数据：{len(bar_data)} 根 K 线")


def calculate_psr(returns, benchmark_sr=0):
    """计算 Probabilistic Sharpe Ratio (PSR)"""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0

    n = len(returns)
    sr = returns.mean() / returns.std(ddof=1) if returns.std() != 0 else 0
    skew = returns.skew()
    kurt = returns.kurtosis() + 3

    sigma_sr = np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1))
    psr = norm.cdf((sr - benchmark_sr) / sigma_sr) if sigma_sr > 0 else 0.5
    return psr


def calculate_dsr(returns, n_trials, sr_std):
    """计算 Deflated Sharpe Ratio (DSR)"""
    if n_trials <= 0:
        n_trials = 1

    ppf_val = norm.ppf(1 - 1/n_trials) if n_trials > 1 else 0
    expected_max_sr = sr_std * (
        (1 - 0.5772156649) * ppf_val +
        0.5772156649 * norm.ppf(1 - 1/n_trials * np.exp(-1)) if n_trials > 1 else 0
    )

    dsr = calculate_psr(returns, benchmark_sr=expected_max_sr)
    return dsr, expected_max_sr


# 步骤选择
step = st.radio(
    "选择步骤",
    ["1. 策略配置", "2. 回测运行", "3. 绩效指标", "4. 可视化分析"],
    horizontal=True
)

if step == "1. 策略配置":
    st.markdown("### 1️⃣ 策略配置")

    st.markdown("#### 交易参数")

    col1, col2, col3 = st.columns(3)

    with col1:
        initial_capital = st.number_input(
            "初始资金",
            min_value=10000,
            value=1000000,
            step=10000
        )

    with col2:
        commission = st.number_input(
            "手续费率",
            min_value=0.0,
            max_value=0.01,
            value=0.001,
            step=0.0001,
            help="每次交易的手续费比例"
        )

    with col3:
        slippage = st.number_input(
            "滑点",
            min_value=0.0,
            max_value=0.01,
            value=0.0005,
            step=0.0001,
            help="价格滑点估计"
        )

    st.markdown("#### 仓位管理")

    bet_size_method = st.selectbox(
        "赌注大小方法",
        ["fixed", "Kelly", "half_Kelly", "custom"],
        format_func=lambda x: {
            'fixed': '固定仓位',
            'Kelly': 'Kelly 公式',
            'half_Kelly': '半 Kelly',
            'custom': '自定义'
        }.get(x, x)
    )

    if bet_size_method == "fixed":
        fixed_size = st.slider("固定仓位比例", 0.1, 1.0, 0.5)
    elif bet_size_method == "custom":
        custom_size = st.slider("自定义仓位", 0.0, 1.0, 0.5)

    st.markdown("#### DSR 参数")

    col1, col2 = st.columns(2)

    with col1:
        n_trials = st.number_input(
            "试验次数",
            min_value=1,
            max_value=1000,
            value=50,
            help="策略开发过程中的试验次数"
        )

    with col2:
        sr_std = st.number_input(
            "Sharpe 标准差估计",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1,
            help="Sharpe 比率的标准差估计"
        )

    # 保存配置
    if st.button("保存回测配置"):
        backtest_config = {
            'initial_capital': initial_capital,
            'commission': commission,
            'slippage': slippage,
            'bet_size_method': bet_size_method,
            'fixed_size': fixed_size if bet_size_method == "fixed" else 0.5,
            'n_trials': n_trials,
            'sr_std': sr_std
        }
        SessionManager.update('backtest_config', backtest_config)
        st.success("回测配置已保存")
        st.json(backtest_config)

elif step == "2. 回测运行":
    st.markdown("### 2️⃣ 回测运行")

    backtest_config = SessionManager.get('backtest_config', {})

    if not backtest_config:
        st.warning("请先配置回测参数")
    else:
        st.json(backtest_config)

        if st.button("运行回测"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # 准备数据
                status_text.text("准备数据...")
                df = bar_data.copy()

                if len(df) < 10:
                    st.error("数据量不足，至少需要 10 根 K 线")
                    st.stop()

                progress_bar.progress(10)

                # 如果有模型，使用模型预测
                model = SessionManager.get('model')
                features = SessionManager.get('features')

                if model is not None and features is not None:
                    status_text.text("生成预测信号...")

                    # 准备特征
                    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
                    feature_cols = [c for c in numeric_cols if c not in ['open', 'high', 'low', 'close', 'volume']]

                    X = features[feature_cols].dropna()

                    # 获取预测
                    y_pred = model.predict(X)
                    y_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None

                    # 映射到信号
                    if y_proba is not None and y_proba.shape[1] > 1:
                        # 使用概率作为信号强度
                        signal = y_proba[:, -1] - y_proba[:, 0]  # P(1) - P(-1)
                    else:
                        signal = y_pred.astype(float)

                    # 保存到 df
                    common_idx = df.index.intersection(X.index)
                    df.loc[common_idx, 'signal'] = signal

                    progress_bar.progress(30)
                else:
                    # 简单动量策略作为默认
                    status_text.text("使用默认动量策略...")
                    df['signal'] = np.sign(df['close'].pct_change().shift(1))
                    progress_bar.progress(30)

                # 应用赌注大小
                status_text.text("计算仓位...")
                bet_method = backtest_config.get('bet_size_method', 'fixed')
                fixed_size = backtest_config.get('fixed_size', 0.5)

                if bet_method == 'fixed':
                    df['position'] = df['signal'] * fixed_size
                else:
                    df['position'] = df['signal'] * 0.5  # 简化

                df['position'] = df['position'].fillna(0)
                progress_bar.progress(50)

                # 计算收益
                status_text.text("计算收益...")
                df['ret'] = df['close'].pct_change()

                # 策略收益 (滞后 1 期避免前视偏差)
                df['strat_ret'] = df['position'].shift(1) * df['ret']

                # 扣除手续费和滑点
                commission = backtest_config.get('commission', 0.001)
                slippage = backtest_config.get('slippage', 0.0005)

                # 交易成本
                position_change = df['position'].diff().abs()
                df['cost'] = position_change * (commission + slippage)
                df['strat_ret'] = df['strat_ret'] - df['cost'].fillna(0)
                df['strat_ret'] = df['strat_ret'].fillna(0)

                progress_bar.progress(70)

                # 计算权益曲线
                status_text.text("计算权益曲线...")
                initial_capital = backtest_config.get('initial_capital', 1000000)

                df['cum_ret'] = (1 + df['strat_ret']).cumprod()
                df['equity'] = initial_capital * df['cum_ret']

                # 计算回撤
                rolling_max = df['equity'].cummax()
                df['drawdown'] = (df['equity'] - rolling_max) / rolling_max

                progress_bar.progress(90)

                # 保存结果
                SessionManager.update('backtest_results', {
                    'equity': df['equity'],
                    'drawdown': df['drawdown'],
                    'returns': df['strat_ret'],
                    'position': df['position'],
                    'signal': df['signal']
                })
                SessionManager.update('backtest_df', df)

                status_text.text("回测完成!")
                progress_bar.progress(100)

                st.success("✅ 回测运行完成")

            except Exception as e:
                st.error(f"回测失败：{str(e)}")
                import traceback
                st.code(traceback.format_exc())

elif step == "3. 绩效指标":
    st.markdown("### 3️⃣ 绩效指标")

    backtest_results = SessionManager.get('backtest_results')

    if backtest_results is None:
        st.warning("请先运行回测")
    else:
        df = SessionManager.get('backtest_df')
        backtest_config = SessionManager.get('backtest_config', {})

        returns = backtest_results['returns']
        equity = backtest_results['equity']
        drawdown = backtest_results['drawdown']

        st.markdown("#### 核心指标")

        # 总收益
        total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1 if len(equity) > 0 else 0

        # 年化 Sharpe
        avg_delta = df.index.to_series().diff().mean() if hasattr(df.index, 'to_series') else pd.Timedelta(minutes=1)
        annualization_factor = (pd.Timedelta(days=365) / avg_delta) if avg_delta and avg_delta.total_seconds() > 0 else 252

        sr = returns.mean() / returns.std(ddof=1) * np.sqrt(annualization_factor) if returns.std() != 0 else 0

        # 最大回撤
        max_dd = drawdown.min()

        # 计算胜率
        winning_trades = (returns > 0).sum()
        total_trades = len(returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("总收益", f"{total_ret:.2%}")

        with col2:
            st.metric("Sharpe 比率", f"{sr:.2f}")

        with col3:
            st.metric("最大回撤", f"{max_dd:.2%}")

        with col4:
            st.metric("胜率", f"{win_rate:.1%}")

        st.markdown("#### 高级指标 (PSR/DSR)")

        n_trials = backtest_config.get('n_trials', 50)
        sr_std = backtest_config.get('sr_std', 0.5)

        psr = calculate_psr(returns)
        dsr, exp_max_sr = calculate_dsr(returns, n_trials, sr_std)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("PSR (SR>0)", f"{psr:.4f}")

        with col2:
            st.metric("预期最大 Sharpe", f"{exp_max_sr:.2f}")

        with col3:
            st.metric("DSR", f"{dsr:.4f}")

        # 结论
        if dsr > 0.95:
            st.success(f"✅ ACCEPT - DSR={dsr:.4f} > 0.95，策略显著优于随机")
        elif dsr > 0.90:
            st.info(f"⚠️ MARGINAL - DSR={dsr:.4f}，策略边缘显著，需要更多验证")
        else:
            st.error(f"❌ REJECT - DSR={dsr:.4f} < 0.95，策略可能是数据挖掘的结果")

        # 详细统计
        st.markdown("#### 详细统计")

        stats = {
            "样本数": len(returns),
            "年化因子": f"{annualization_factor:.1f}",
            "日均收益": f"{returns.mean():.6f}",
            "收益标准差": f"{returns.std():.6f}",
            "偏度": f"{returns.skew():.4f}",
            "峰度": f"{returns.kurtosis():.4f}",
            "正收益天数": int(winning_trades),
            "负收益天数": int(total_trades - winning_trades),
        }

        st.json(stats)

elif step == "4. 可视化分析":
    st.markdown("### 4️⃣ 可视化分析")

    backtest_results = SessionManager.get('backtest_results')

    if backtest_results is None:
        st.warning("请先运行回测")
    else:
        equity = backtest_results['equity']
        drawdown = backtest_results['drawdown']
        position = backtest_results.get('position')

        # 权益曲线
        st.markdown("#### 权益曲线")

        fig = render_equity_curve(equity, title="策略权益曲线")
        st.plotly_chart(fig, use_container_width=True)

        # 回撤图
        st.markdown("#### 回撤分析")

        fig_dd = render_drawdown_chart(drawdown, title="回撤走势")
        st.plotly_chart(fig_dd, use_container_width=True)

        # 仓位变化
        if position is not None:
            st.markdown("#### 仓位变化")

            fig_pos = render_line_chart(
                pd.DataFrame({'position': position}),
                ['position'],
                title="仓位变化"
            )
            st.plotly_chart(fig_pos, use_container_width=True)

        # 收益分布
        st.markdown("#### 收益分布")

        from components.charts import render_distribution_chart
        returns = backtest_results['returns']

        fig_dist = render_distribution_chart(returns.dropna().values, title="策略收益分布")
        st.plotly_chart(fig_dist, use_container_width=True)

# 导航按钮
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("⬅️ 返回模型训练", use_container_width=True):
        navigate_to("5️⃣ 模型训练")
        st.rerun()

with col3:
    if st.button("前往可视化中心 🎨", use_container_width=True):
        navigate_to("🎨 可视化中心")
        st.rerun()
