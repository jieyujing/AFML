# CSV 数据支持与 Dollar Bar 生成实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 AFMLKit Web UI 添加 CSV 数据加载和 1 分钟 Dollar Bar 生成功能

**Architecture:** 复用现有的 Dollar Bar 生成逻辑（evaluate_dollar_bars.py），封装为 Web UI 兼容的流式处理模块，通过 Streamlit 页面提供交互界面

**Tech Stack:** Streamlit, Pandas, NumPy, Numba, Plotly

---

## 任务列表

### 任务 1: 创建 CSV 加载器工具模块

**Files:**
- Create: `webapp/utils/csv_loader.py`
- Test: `tests/webapp/test_csv_loader.py`

**步骤:**
1. 创建 `webapp/utils/csv_loader.py` 文件
2. 实现 `CSVDataloader` 类，包含以下方法:
   - `__init__(data_dir: str)` - 初始化数据目录
   - `list_available_files() -> List[Path]` - 列出可用 CSV 文件
   - `load_file(filepath: str, column_mapping: Dict = None) -> pd.DataFrame` - 加载文件
   - `validate_ohlcv(df: pd.DataFrame) -> Tuple[bool, List[str]]` - 验证格式
3. 支持列名自动识别（datetime, open, high, low, close, volume）
4. 编写单元测试
5. 运行测试验证

---

### 任务 2: 创建 Dollar Bar 生成器模块

**Files:**
- Create: `webapp/utils/dollar_bar_generator.py`

**步骤:**
1. 创建 `webapp/utils/dollar_bar_generator.py` 文件
2. 从 `scripts/evaluate_dollar_bars.py` 提取核心逻辑:
   - `_stream_dynamic_dollar_bar_indexer` (Numba 函数)
   - `_build_ohlcv_from_indices`
3. 实现 `DollarBarGenerator` 类:
   - `__init__(target_daily_bars, ewma_span)` - 初始化参数
   - `scan_daily_dollar_volume(df) -> pd.Series` - 计算日度 dollar volume
   - `compute_ewma_thresholds(daily_vol) -> Dict[int, pd.Series]` - EWMA 阈值
   - `generate_bars(df, thresholds) -> Dict[int, pd.DataFrame]` - 生成 bars
   - `save_bars(bars_dict, output_dir)` - 保存 CSV
4. 添加进度回调支持

---

### 任务 3: 创建 IID 评估器模块

**Files:**
- Create: `webapp/utils/iid_evaluator.py`

**步骤:**
1. 创建 `webapp/utils/iid_evaluator.py` 文件
2. 实现 `IIDEvaluator` 类:
   - `evaluate(ohlcv) -> Dict` - 执行完整评估
   - `jarque_bera_test(returns) -> Tuple[float, float]` - JB 检验
   - `autocorrelation(returns, lag) -> float` - 自相关计算
   - `score(results) -> int` - 综合评分，返回最优频率
3. 添加统计指标计算（偏度、峰度、均值、标准差）

---

### 任务 4: 创建 Dollar Bar 可视化组件

**Files:**
- Create: `webapp/components/dollar_bar_viz.py`

**步骤:**
1. 创建 `webapp/components/dollar_bar_viz.py` 文件
2. 实现以下可视化函数:
   - `plot_iid_comparison(results, best_freq) -> go.Figure` - IID 指标对比
   - `plot_return_distribution(returns, best_freq) -> go.Figure` - 收益分布
   - `plot_bars_count_comparison(bars_dict) -> go.Figure` - Bar 数量对比
   - `plot_time_series(bars_dict, best_freq) -> go.Figure` - 时间序列对比
3. 使用 Plotly 交互式图表
4. 支持自定义颜色和样式

---

### 任务 5: 修改数据导入页面添加 CSV 支持

**Files:**
- Modify: `webapp/pages/01_data_import.py`

**步骤:**
1. 导入 `CSVDataloader`
2. 在数据源选项中添加 "从 data/csv 目录选择"
3. 实现文件列表下拉框
4. 添加列自动识别和映射功能
5. 添加数据预览和验证反馈

---

### 任务 6: 创建 Dollar Bar 生成页面

**Files:**
- Create: `webapp/pages/02_dollar_bar_generator.py`

**步骤:**
1. 创建新页面文件
2. 实现 UI 组件:
   - 数据文件选择器
   - 参数配置（目标频率多选、EWMA Span）
   - 运行按钮和进度条
   - 结果展示区域
3. 集成 `DollarBarGenerator` 和 `IIDEvaluator`
4. 添加可视化展示
5. 添加 CSV 导出按钮
6. 在 `components/sidebar.py` 中注册页面

---

### 任务 7: 更新侧边栏导航

**Files:**
- Modify: `webapp/components/sidebar.py`

**步骤:**
1. 在 `PAGES` 字典中添加新页面:
   ```python
   "2️⃣ Dollar Bar": {
       "icon": "💵",
       "file": "pages/02_dollar_bar_generator.py",
       "description": "生成和评估 Dollar Bars"
   }
   ```
2. 调整后续页面编号

---

### 任务 8: 集成测试与验证

**Files:**
- Test: 使用 `data/csv/Y9999.XDCE-2023-1-1-To-2026-03-11-1m.csv`

**步骤:**
1. 启动 Web UI: `streamlit run webapp/app.py`
2. 测试从 data/csv 加载数据
3. 测试 Dollar Bar 生成（使用默认参数 [4, 6, 10, 20, 50]）
4. 验证 IID 评估结果
5. 验证可视化渲染
6. 验证 CSV 导出功能
7. 记录并修复发现的问题

---

### 任务 9: 文档更新

**Files:**
- Modify: `webapp/README.md`

**步骤:**
1. 更新功能特性列表
2. 添加 Dollar Bar 生成使用说明
3. 添加示例截图
4. 提交文档

---

## 依赖关系

```
任务 1 → 任务 5
任务 2 → 任务 6
任务 3 → 任务 6
任务 4 → 任务 6
任务 5, 6, 7 → 任务 8
任务 8 → 任务 9
```

## 验收标准

- [ ] 可以从 `data/csv` 目录选择并加载 CSV 文件
- [ ] 可以上传外部 CSV 文件
- [ ] 正确生成 5 种频率的 Dollar Bars
- [ ] IID 评估包含 JB 统计量、p-value、自相关
- [ ] 最优频率推荐功能正常
- [ ] 4 个可视化图表正确渲染
- [ ] CSV 导出功能正常
- [ ] 所有测试通过
