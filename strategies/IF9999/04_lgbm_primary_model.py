"""
04_lgbm_primary_model.py - IF9999 LightGBM Primary Model

使用 LightGBM 预测 CUSUM 事件点的收益方向：
1. 加载事件特征 (events_features.parquet)
2. 使用 TBM 收益方向作为标签
3. Purged K-Fold CV 训练
4. 输出 side 信号 (+1/-1/0)
5. TBM 回测验证

AFML 方法论：
- Primary Model 目标：高 Recall（不错过机会）
- 使用 Purged CV 防止信息泄露
- 特征平稳性已通过 FracDiff 处理

评估指标：
- Recall (关键)：不错过机会
- Precision：控制假信号
- 胜率：> 52%
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, classification_report, confusion_matrix
)

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.IF9999.config import (
    BARS_DIR, FIGURES_DIR, FEATURES_DIR,
    TBM_CONFIG
)

# 定义模型输出目录
MODELS_DIR = os.path.join(os.path.dirname(FEATURES_DIR), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
from afmlkit.validation import PurgedKFold

# ============================================================
# 配置参数
# ============================================================

LGBM_CONFIG = {
    # 模型参数
    'n_estimators': 500,
    'max_depth': 5,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,

    # 信号生成
    'probability_threshold': 0.5,  # 概率阈值
    'signal_margin': 0.1,          # 信号边界（控制信号稀疏性）

    # CV 参数
    'cv_n_splits': 5,
    'cv_embargo_pct': 0.01,
}

# 排除的特征列（含未来信息或非特征）
EXCLUDE_COLS = [
    'open', 'high', 'low', 'close', 'volume', 'vwap',
    'trend_scan_side', 'trend_scan_t_value', 'trend_scan_window',
    'trend_labels', 'side', 'label', 'ret', 'pnl', 'touch_type'
]


# ============================================================
# 数据加载
# ============================================================

def load_data():
    """加载所有需要的数据。"""
    print("\n[Step 1] 加载数据...")

    # 加载 Dollar Bars
    bars_path = os.path.join(BARS_DIR, 'dollar_bars_target6.parquet')
    bars = pd.read_parquet(bars_path)
    print(f"  Dollar Bars: {len(bars)} bars")
    print(f"  时间范围: {bars.index.min()} ~ {bars.index.max()}")

    # 加载事件特征
    features_path = os.path.join(FEATURES_DIR, 'events_features.parquet')
    events_features = pd.read_parquet(features_path)
    print(f"  事件特征: {events_features.shape[0]} 样本 × {events_features.shape[1]} 特征")

    # 加载 TBM 结果（作为标签来源）
    tbm_path = os.path.join(FEATURES_DIR, 'tbm_results.parquet')
    if not os.path.exists(tbm_path):
        print(f"  ❌ TBM 结果不存在，请先运行 04_ma_primary_model.py")
        return None, None, None, None
    tbm = pd.read_parquet(tbm_path)
    print(f"  TBM 结果: {len(tbm)} 样本")

    # 加载 CUSUM 事件
    cusum_path = os.path.join(FEATURES_DIR, 'cusum_events.parquet')
    cusum_events = pd.read_parquet(cusum_path)
    print(f"  CUSUM 事件: {len(cusum_events)} 个")

    return bars, events_features, tbm, cusum_events


# ============================================================
# 特征准备
# ============================================================

def prepare_features_labels(events_features, tbm):
    """
    准备特征矩阵和标签。

    标签定义：TBM 收益方向
    - ret > 0 → y = 1 (正确预测)
    - ret < 0 → y = 0 (错误预测)
    """
    print("\n[Step 2] 准备特征和标签...")

    # 对齐索引
    common_idx = events_features.index.intersection(tbm.index)
    if len(common_idx) == 0:
        raise ValueError("事件特征和 TBM 结果没有交集！")

    X = events_features.loc[common_idx].copy()
    tbm_aligned = tbm.loc[common_idx].copy()

    # 标签：收益方向
    y = (tbm_aligned['ret'] > 0).astype(int)

    # 结束时间（用于 Purging）
    t1 = tbm_aligned['exit_ts']

    # 排除非特征列
    feature_cols = [c for c in X.columns
                    if c not in EXCLUDE_COLS and not c.startswith('trend_')]
    X = X[feature_cols]

    # 处理缺失值
    X = X.fillna(0)

    # 处理无穷值
    X = X.replace([np.inf, -np.inf], 0)

    print(f"  特征数: {X.shape[1]}")
    print(f"  样本数: {len(X)}")
    print(f"  标签分布: 1={y.sum()}, 0={len(y)-y.sum()}")
    print(f"  正向比例: {y.mean()*100:.1f}%")

    return X, y, t1, tbm_aligned


# ============================================================
# LightGBM 模型训练
# ============================================================

def train_lgbm_model(X, y, t1, config):
    """
    使用 Purged K-Fold CV 训练 LightGBM 模型。

    返回 OOF 预测和训练好的模型。
    """
    print("\n[Step 3] 训练 LightGBM 模型...")
    print(f"  CV: {config['cv_n_splits']}-Fold Purged CV (embargo={config['cv_embargo_pct']:.1%})")

    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        print("  ❌ LightGBM 未安装，正在安装...")
        import subprocess
        subprocess.run(['uv', 'add', 'lightgbm'], check=True)
        from lightgbm import LGBMClassifier

    # 初始化模型
    model_params = {k: v for k, v in config.items()
                    if k not in ['probability_threshold', 'signal_margin', 'cv_n_splits', 'cv_embargo_pct']}
    model = LGBMClassifier(**model_params)

    # Purged K-Fold CV
    cv = PurgedKFold(
        n_splits=config['cv_n_splits'],
        t1=t1,
        embargo_pct=config['cv_embargo_pct']
    )

    # OOF 预测
    oof_pred = np.zeros(len(X))
    oof_prob = np.zeros(len(X))

    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 训练
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        oof_pred[test_idx] = y_pred
        oof_prob[test_idx] = y_prob

        # 计算指标
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        fold_metrics.append({
            'fold': fold + 1,
            'f1': f1,
            'precision': prec,
            'recall': rec,
            'accuracy': acc,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
        })

        print(f"  Fold {fold+1}: F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, Acc={acc:.4f}")

    # 汇总
    metrics_df = pd.DataFrame(fold_metrics)
    print(f"\n[CV 结果汇总]")
    print(f"  F1-Score: {metrics_df['f1'].mean():.4f} ± {metrics_df['f1'].std():.4f}")
    print(f"  Precision: {metrics_df['precision'].mean():.4f} ± {metrics_df['precision'].std():.4f}")
    print(f"  Recall: {metrics_df['recall'].mean():.4f} ± {metrics_df['recall'].std():.4f}")
    print(f"  Accuracy: {metrics_df['accuracy'].mean():.4f} ± {metrics_df['accuracy'].std():.4f}")

    # 在完整数据集上重新训练
    print("\n  在完整数据集上重新训练最终模型...")
    model.fit(X, y)

    return model, oof_pred, oof_prob, metrics_df


# ============================================================
# 信号生成
# ============================================================

def generate_signals(oof_prob, config):
    """
    生成交易信号。

    信号规则：
    - prob > threshold + margin → long (+1)
    - prob < threshold - margin → short (-1)
    - 其他 → 无信号 (0)
    """
    print("\n[Step 4] 生成交易信号...")

    threshold = config['probability_threshold']
    margin = config['signal_margin']

    signals = np.zeros(len(oof_prob))
    signals[oof_prob > threshold + margin] = 1   # Long
    signals[oof_prob < threshold - margin] = -1  # Short

    n_long = (signals == 1).sum()
    n_short = (signals == -1).sum()
    n_neutral = (signals == 0).sum()

    print(f"  信号分布:")
    print(f"    Long (+1): {n_long} ({n_long/len(signals)*100:.1f}%)")
    print(f"    Short (-1): {n_short} ({n_short/len(signals)*100:.1f}%)")
    print(f"    Neutral (0): {n_neutral} ({n_neutral/len(signals)*100:.1f}%)")

    return signals.astype(int)


# ============================================================
# TBM 回测
# ============================================================

def run_tbm_backtest(signals, tbm_aligned):
    """
    使用 TBM 回测生成的信号。
    """
    print("\n[Step 5] TBM 回测...")

    # 过滤无信号
    valid_mask = signals != 0
    valid_signals = signals[valid_mask]
    valid_tbm = tbm_aligned.iloc[valid_mask]

    if len(valid_signals) == 0:
        print("  ⚠️ 无有效信号")
        return None

    # 计算收益
    returns = pd.Series(valid_tbm['ret'].values * valid_signals, index=valid_tbm.index)

    # 统计
    n_total = len(returns)
    n_win = (returns > 0).sum()
    n_loss = (returns < 0).sum()
    win_rate = n_win / n_total if n_total > 0 else 0

    avg_ret = returns.mean()
    total_ret = returns.sum()

    avg_win = returns[returns > 0].mean() if n_win > 0 else 0
    avg_loss = returns[returns < 0].mean() if n_loss > 0 else 0
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    stats = {
        'n_total': n_total,
        'n_win': n_win,
        'n_loss': n_loss,
        'win_rate': win_rate,
        'avg_ret': avg_ret,
        'total_ret': total_ret,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_loss_ratio': profit_loss_ratio,
        'returns': returns,
    }

    print(f"  总信号数: {n_total}")
    print(f"  胜率: {win_rate*100:.1f}%")
    print(f"  平均收益: {avg_ret*100:.4f}%")
    print(f"  盈亏比: {profit_loss_ratio:.2f}")

    return stats


# ============================================================
# 可视化
# ============================================================

def plot_results(oof_prob, y, signals, stats, output_dir):
    """生成可视化图表。"""
    print("\n[Step 6] 生成可视化图表...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. 概率分布
    ax = axes[0, 0]
    ax.hist(oof_prob[y == 0], bins=50, alpha=0.7, label='Label 0', color='red')
    ax.hist(oof_prob[y == 1], bins=50, alpha=0.7, label='Label 1', color='green')
    ax.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    ax.set_xlabel('Probability')
    ax.set_ylabel('Count')
    ax.set_title('OOF Probability Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 累积收益
    ax = axes[0, 1]
    if stats and stats['returns'] is not None:
        cum_ret = (1 + stats['returns']).cumprod()
        cum_ret.plot(ax=ax)
        ax.set_xlabel('Time')
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Cumulative Returns')
        ax.grid(True, alpha=0.3)

    # 3. 信号分布
    ax = axes[1, 0]
    signal_counts = pd.Series(signals).value_counts()
    colors = ['red' if x == -1 else 'gray' if x == 0 else 'green' for x in signal_counts.index]
    ax.bar(signal_counts.index.map({-1: 'Short', 0: 'Neutral', 1: 'Long'}),
           signal_counts.values, color=colors)
    ax.set_xlabel('Signal')
    ax.set_ylabel('Count')
    ax.set_title('Signal Distribution')
    ax.grid(True, alpha=0.3)

    # 4. 收益分布
    ax = axes[1, 1]
    if stats and stats['returns'] is not None:
        ax.hist(stats['returns'], bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--')
        ax.set_xlabel('Return')
        ax.set_ylabel('Count')
        ax.set_title(f'Return Distribution (Win Rate: {stats["win_rate"]*100:.1f}%)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig_path = os.path.join(output_dir, '04_lgbm_primary_model.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  ✅ 图表已保存: {fig_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    """LightGBM Primary Model 主流程。"""
    print("=" * 70)
    print("  IF9999 LightGBM Primary Model")
    print("=" * 70)

    # 确保输出目录存在
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Step 1: 加载数据
    bars, events_features, tbm, cusum_events = load_data()
    if bars is None:
        return

    # Step 2: 准备特征和标签
    X, y, t1, tbm_aligned = prepare_features_labels(events_features, tbm)

    # Step 3: 训练模型
    model, oof_pred, oof_prob, metrics_df = train_lgbm_model(X, y, t1, LGBM_CONFIG)

    # Step 4: 生成信号
    signals = generate_signals(oof_prob, LGBM_CONFIG)

    # Step 5: TBM 回测
    stats = run_tbm_backtest(signals, tbm_aligned)

    # Step 6: 可视化
    plot_results(oof_prob, y, signals, stats, FIGURES_DIR)

    # Step 7: 保存结果
    print("\n[Step 7] 保存结果...")

    # 保存信号
    signals_df = pd.DataFrame({
        'side': signals,
        'prob': oof_prob,
        'y_true': y,
    }, index=X.index)
    signals_path = os.path.join(FEATURES_DIR, 'lgbm_primary_signals.parquet')
    signals_df.to_parquet(signals_path)
    print(f"  ✅ 信号已保存: {signals_path}")

    # 保存模型
    import joblib
    model_path = os.path.join(MODELS_DIR, 'lgbm_primary_model.pkl')
    joblib.dump(model, model_path)
    print(f"  ✅ 模型已保存: {model_path}")

    # 保存特征重要性
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    importance_path = os.path.join(MODELS_DIR, 'lgbm_feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"  ✅ 特征重要性已保存: {importance_path}")

    # Step 8: 输出总结
    print("\n" + "=" * 70)
    print("  LightGBM Primary Model 结果总结")
    print("=" * 70)

    avg_f1 = metrics_df['f1'].mean()
    avg_recall = metrics_df['recall'].mean()

    print(f"\n模型性能:")
    print(f"  F1-Score: {avg_f1:.4f}")
    print(f"  Recall: {avg_recall:.4f}")
    print(f"  Precision: {metrics_df['precision'].mean():.4f}")

    if stats:
        print(f"\n回测结果:")
        print(f"  胜率: {stats['win_rate']*100:.1f}%")
        print(f"  平均收益: {stats['avg_ret']*100:.4f}%")
        print(f"  盈亏比: {stats['profit_loss_ratio']:.2f}")

    print(f"\nTop 5 重要特征:")
    for i, row in importance_df.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']}")

    # 评估结论
    print("\n" + "-" * 70)
    if stats and stats['win_rate'] > 0.53:
        print("  ✅ LightGBM Primary Model 表现优于 MA (胜率 > 53%)")
    elif stats and stats['win_rate'] > 0.51:
        print("  ⚠️ LightGBM Primary Model 表现接近 MA (胜率 ~51%)")
    else:
        print("  ❌ LightGBM Primary Model 表现不佳 (胜率 < 51%)")
    print("-" * 70)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()