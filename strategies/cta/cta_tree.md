单品种 CTA / AFML：一张网过一次筛

0. 原始市场流
   ├── 网：原始逐笔 / K线行情
   └── 筛：dollar bars
       ├── 目的：把自然时间变成交易活跃度时间
       └── 验证：
           ├── bar_count
           ├── median_bar_duration
           ├── dollar_value_per_bar 分布
           └── 缺失 / 重复 / 异常值检查

1. 市场时点选择
   ├── 网：全部 dollar bars
   └── 筛：CUSUM filter
       ├── 目的：只保留累计偏移足够大的时点
       ├── 输出：event timestamps
       ├── 推荐搜索空间：
       │   └── 目标事件率：5% / 10% / 15%
       └── 验证：
           ├── events_per_1000_bars
           ├── bars_between_events
           ├── event_overlap_ratio
           └── event 后波动 / 方向性是否高于随机时点

2. 方向候选生成
   ├── 网：所有 CUSUM events
   └── 筛：DMA（双均线）
       ├── 目的：给每个 event 一个粗方向
       ├── 输出：long candidate / short candidate
       ├── 推荐搜索空间：
       │   ├── (5,20)
       │   ├── (5,30)
       │   ├── (10,30)
       │   ├── (10,40)
       │   ├── (15,40)
       │   └── (20,60)
       └── 验证：
           ├── 信号数
           ├── 多空占比
           └── 与 event 的对齐是否稳定

3. 趋势真实性辅助验证
   ├── 网：所有 DMA candidate signals
   └── 筛：Trend scanning
       ├── 目的：检查 DMA 方向后面是否真的出现过一段可辨认趋势
       ├── 输出：
       │   ├── trend side
       │   ├── best window
       │   └── t-value / strength
       ├── 推荐搜索空间：
       │   ├── windows family A: [5, 10, 20, 40]
       │   ├── windows family B: [10, 20, 30, 50]
       │   └── 强趋势阈值：|t| > 1.5 / |t| > 2.0
       └── 验证：
           ├── |t| 分布
           ├── best_window 分布
           ├── signed_t = DMA_side * trend_scan_tvalue
           ├── trend_alignment_rate
           └── high_t_recall

4. Direction 层正式验证
   ├── 网：所有 DMA signals + trend scan 结果
   └── 筛：signed_t 指标体系
       ├── 目的：验证 primary 是否抓到“对方向的趋势机会”
       └── 验证：
           ├── mean(signed_t)
           ├── median(signed_t)
           ├── p75 / p90 of signed_t
           ├── share(signed_t > 1.5)
           ├── share(signed_t > 2.0)
           ├── share(signed_t < -1.5)
           └── 不同 DMA pair 下是否 broadly consistent

5. 条件异质性检查
   ├── 网：所有 DMA signals（含 trend scan 标签）
   └── 筛：condition features
       ├── 目的：看同样的 DMA 信号，是否因为条件不同而质量不同
       ├── 推荐特征（起步 5 个）：
       │   ├── whipsaw_count
       │   ├── trend_efficiency
       │   ├── ATR_percentile
       │   ├── compression_score
       │   └── recent_false_breakout_count
       ├── 推荐标签：
       │   ├── 回归：y = DMA_side * trend_scan_tvalue
       │   └── 三分类：
       │       ├── +1 = 方向一致且 |t| > 2.0
       │       ├──  0 = |t| <= 1.5
       │       └── -1 = 方向相反且 |t| > 2.0
       └── 验证：
           ├── 单特征分桶（3桶 / 5桶）
           ├── mean(y) / median(y)
           ├── 强趋势占比
           ├── top bucket vs bottom bucket
           ├── rank IC / Spearman
           └── 是否存在单调性

6. Meta / quality score（可选）
   ├── 网：已确认存在条件异质性的 signals
   └── 筛：meta model / rank score
       ├── 目的：把条件判断变成质量分数
       ├── 推荐搜索空间：
       │   ├── 先做分桶
       │   ├── Logistic / ordinal logistic
       │   └── LightGBM（浅树）
       │       ├── max_depth: 2 / 3 / 4
       │       ├── n_estimators: 50 / 100 / 200
       │       ├── learning_rate: 0.03 / 0.05 / 0.1
       │       └── min_child_samples: 30 / 50 / 100
       └── 验证：
           ├── AUC / macro-F1（三分类时）
           ├── IC / rank IC（回归时）
           ├── top20% - bottom20% 差值
           ├── bucket monotonicity
           └── 不同年份下是否稳定

7. Base sizing
   ├── 网：所有通过 quality 筛选的 signals
   └── 筛：vol targeting
       ├── 目的：统一风险单位
       ├── 公式：base_size = target_risk / realized_vol
       └── 验证：
           ├── 仓位是否过于跳跃
           ├── 波动高时是否自然减仓
           └── 单笔风险是否受控

8. Quality-adjusted sizing
   ├── 网：所有已有 base_size 的 signals
   └── 筛：quality mapping
       ├── 目的：高质量信号拿到更多资本
       ├── 推荐搜索空间：
       │   ├── 二档：
       │   │   ├── bottom 50% = 0
       │   │   └── top 50% = 1
       │   ├── 三档：
       │   │   ├── bottom 40% = 0
       │   │   ├── middle 40% = 0.5
       │   │   └── top 20% = 1
       │   └── 分位数线性：
       │       └── size_multiplier = rank_pct(quality)
       └── 验证：
           ├── coverage
           ├── avg_pnl_per_trade
           ├── total_pnl
           ├── payoff_ratio
           ├── Sharpe / Calmar
           ├── MDD
           └── coverage-quality curve

9. 单信号 Exit 兑现
   ├── 网：每一个独立 signal 的生命周期
   └── 筛：exit family
       ├── 目的：把纸面趋势机会兑现成 realized pnl
       ├── 推荐搜索空间：
       │   ├── 固定持有期：H = 10 / 20
       │   ├── ATR stop + time stop：
       │   │   ├── SL = 1.0 ATR, T = 20
       │   │   └── SL = 1.5 ATR, T = 20
       │   └── opposite DMA signal exit
       ├── 每个 signal 单独跟踪：
       │   ├── entry_time / exit_time
       │   ├── entry_price / exit_price
       │   ├── holding_bars
       │   ├── MFE / MAE
       │   ├── realized_pnl
       │   ├── trend_scan_t
       │   ├── best_window
       │   └── is_overlapped / n_active_signals
       └── 验证：
           ├── mean / median trade pnl
           ├── payoff_ratio
           ├── 不同 quality bucket 的兑现差异
           ├── 各 exit family 下是否仍保持单调性
           └── 是否只在单一 exit 下有效

10. 活跃信号聚合（执行表达层）
    ├── 网：所有同时活跃的独立 signals
    └── 筛：aggregation rule
        ├── 目的：从 event-level signals 变成一个净头寸
        ├── 可选规则：
        │   ├── average active bets
        │   ├── max dominance
        │   ├── latest overwrite
        │   └── net summed exposure
        └── 验证：
            ├── 聚合前后收益分布差异
            ├── 是否过度放大重叠信号
            └── 组合头寸是否平滑可执行

11. OOS / 稳健性验证
    ├── 网：所有开发期得到的规则
    └── 筛：时间外验证
        ├── 目的：筛掉只在样本内成立的幻象
        ├── 时间切分：
        │   ├── 研究期：冻结 label family / 特征家族
        │   ├── 开发期：训练与调参
        │   └── 保留期：最终一次性 OOS 检验
        ├── 要冻结的东西：
        │   ├── trend scan windows family
        │   ├── t-threshold
        │   ├── DMA 候选集合
        │   ├── condition 特征集合
        │   ├── sizing mapping 集合
        │   └── exit family 集合
        └── 验证：
            ├── 不同年份下结论是否同向
            ├── trend scan family A/B 下是否同向
            ├── event rate 5/10/15% 下是否 broadly stable
            ├── 成本后是否仍成立
            └── 保留期是否仍有 ranking / bucket separation