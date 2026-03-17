# CUSUM 采样页面设计规范

## 概述

将 CUSUM Filter 功能从"数据导入"页面拆分为独立页面，放置在 Dollar Bar 生成之后、特征工程之前，形成更清晰的数据处理流程。

## 变更范围

### 新增

- 新页面：`webapp/pages/03_cusum_sampling.py`（CUSUM 采样）

### 修改

- `webapp/pages/02_dollar_bar_generator.py` → `webapp/pages/02_dollar_bar.py`（重命名简化）
- `webapp/pages/03_feature_engineering.py` → `webapp/pages/04_feature_engineering.py`（重新编号）
- `webapp/pages/04_labeling.py` → `webapp/pages/05_labeling.py`（重新编号）
- `webapp/pages/05_feature_analysis.py` → `webapp/pages/06_feature_analysis.py`（重新编号）
- `webapp/pages/06_model_training.py` → `webapp/pages/07_model_training.py`（重新编号）
- `webapp/pages/07_backtest.py` → `webapp/pages/08_backtest.py`（重新编号）
- `webapp/pages/08_visualization.py` → `webapp/pages/09_visualization.py`（重新编号）
- `webapp/pages/09_experiment.py` → `webapp/pages/10_experiment.py`（重新编号）
- `webapp/components/sidebar.py`：更新导航菜单，添加新页面
- `webapp/session.py`：添加 CUSUM 相关状态键

### 删除

- 数据导入页面中 `bar_type='cusum'` 相关代码（约 80 行）

## 页面重新编号方案

| 原编号 | 新编号 | 页面名称 |
|--------|--------|----------|
| 01_data_import.py | 01_data_import.py | 数据导入（不变） |
| 02_dollar_bar_generator.py | 02_dollar_bar.py | Dollar Bar（重命名） |
| - | **03_cusum_sampling.py** | **CUSUM 采样（新增）** |
| 03_feature_engineering.py | 04_feature_engineering.py | 特征工程 |
| 04_labeling.py | 05_labeling.py | 标签生成 |
| 05_feature_analysis.py | 06_feature_analysis.py | 特征分析 |
| 06_model_training.py | 07_model_training.py | 模型训练 |
| 07_backtest.py | 08_backtest.py | 回测评估 |
| 08_visualization.py | 09_visualization.py | 可视化中心 |
| 09_experiment.py | 10_experiment.py | 实验管理 |

## 数据流程变更

### 变更前

```
数据导入（含 CUSUM K 线选项）
    ↓
Dollar Bar 生成
    ↓
特征工程（含 CUSUM 对齐选项）
    ↓
标签生成
```

### 变更后

```
数据导入
    ↓
Dollar Bar 生成
    ↓
CUSUM 采样（新页面）
    ↓
特征工程
    ↓
标签生成
```

## 页面设计

### 文件名

`webapp/pages/03_cusum_sampling.py`

### 页面标题

🔬 CUSUM 采样

### 布局结构

单页分组布局，包含 5 个区块：

```
┌─────────────────────────────────────────────────────────────┐
│  🔬 CUSUM 采样                                              │
├─────────────────────────────────────────────────────────────┤
│  📊 数据源                                                  │
│  ⚙️ 参数配置                                                │
│  ▶️ 执行采样                                                │
│  📈 采样结果                                                │
│  💾 导出                                                    │
└─────────────────────────────────────────────────────────────┘
```

### 各区块详细设计

#### 1. 数据源区块

**功能**：选择输入数据来源

**UI 元素**：
- Radio 选择器：`从 Dollar Bars 采样` / `加载已保存结果`
- Dollar Bars 模式：频率下拉选择器（从 session dollar_bars 读取可用频率）
- 加载模式：文件选择器（从 outputs/cusum_sampling/ 读取已有文件）

**状态管理**：
- 输入：`dollar_bars` (Session)
- 输出：`cusum_sampled_data` (Session), `cusum_events` (Session)

**数据验证**：
- 检查 `dollar_bars` 是否包含 `close` 列
- 验证索引为 DatetimeIndex
- 最小数据行数要求：> 100 行（否则显示警告）

**错误处理**：
- 若 `dollar_bars` 为空，显示警告提示并引导用户前往 Dollar Bar 生成页面
- 若加载文件失败，显示错误信息并重置选择器
- 文件损坏时显示具体错误原因

#### 2. 参数配置区块

**功能**：配置 CUSUM 采样参数

**UI 元素**：
- 高级模式 Checkbox（默认关闭）
- `vol_span`：波动率窗口（默认 50，高级模式可调，范围 10-200）
- `threshold_multiplier`：阈值乘数（默认 2.0，高级模式可调，范围 0.5-5.0）
- `use_frac_diff`：启用分数阶差分（默认 True）

**状态管理**：
- 输出：`cusum_config` (Session)

#### 3. 执行采样区块

**功能**：触发采样计算

**UI 元素**：
- 执行按钮
- 进度条（执行中显示）

**状态管理**：
- 输出：`cusum_sampled_data` (Session), `cusum_events` (Session)

**错误处理**：
- 捕获 `compute_dynamic_cusum_filter` 异常，显示错误详情
- 提供重试按钮
- 超时处理：超过 5 分钟显示进度但允许继续

#### 4. 采样结果区块

**功能**：展示采样结果和可视化

**UI 元素**：
- 采样率统计面板（原始行数、采样行数、采样率、时间跨度）
- 价格序列 + 事件标记图（Plotly，复用 cusum_viz.py）
- 数据预览表格（可折叠）

**组件复用**：
- `webapp/components/cusum_viz.py` 中的 `plot_price_with_events()`, `render_sampling_rate_panel()`

#### 5. 导出区块

**功能**：导出采样结果

**UI 元素**：
- 格式选择：CSV / Parquet
- 文件名输入框（默认：`cusum_sampled_YYYYMMDD`）
- 导出按钮

**导出路径**：`outputs/cusum_sampling/`

**导出逻辑**：
- 自动创建 `outputs/cusum_sampling/` 目录
- 若文件已存在，提示覆盖确认（用户确认后才覆盖）
- 导出成功后显示完整路径

## Session 状态键

新增以下状态键到 `SessionManager.KEYS`：

```python
'cusum_config',         # 参数配置
'cusum_sampled_data',   # 采样后数据（DataFrame）
'cusum_events',         # CUSUM 事件时间点数组
```

**说明**：输入数据直接从 `dollar_bars` 读取，无需额外状态键。

## 侧边栏导航更新

### PAGES 字典更新

```python
PAGES = {
    "首页": {...},
    "1️⃣ 数据导入": {
        "icon": "📥",
        "file": "pages/01_data_import.py",
        "description": "导入交易数据并构建 K 线"
    },
    "💵 Dollar Bar": {
        "icon": "💵",
        "file": "pages/02_dollar_bar.py",
        "description": "生成和评估 Dollar Bars"
    },
    "🔬 CUSUM 采样": {  # 新增
        "icon": "🔬",
        "file": "pages/03_cusum_sampling.py",
        "description": "CUSUM 事件采样与可视化"
    },
    "2️⃣ 特征工程": {
        "icon": "🔧",
        "file": "pages/04_feature_engineering.py",
        "description": "构建和变换特征"
    },
    "3️⃣ 标签生成": {...},
    "4️⃣ 特征分析": {...},
    "5️⃣ 模型训练": {...},
    "6️⃣ 回测评估": {...},
    "🎨 可视化中心": {...}
}
```

### 步骤指示器更新

```python
steps = [
    "数据导入",
    "Dollar Bar",
    "CUSUM 采样",  # 新增
    "特征工程",
    "标签生成",
    "模型训练",
    "回测评估"
]
```

### step_mapping 更新

```python
step_mapping = {
    '首页': 0,
    '1️⃣ 数据导入': 0,
    '💵 Dollar Bar': 1,
    '🔬 CUSUM 采样': 2,  # 新增
    '2️⃣ 特征工程': 3,
    '3️⃣ 标签生成': 4,
    '4️⃣ 特征分析': 5,
    '5️⃣ 模型训练': 6,
    '6️⃣ 回测评估': 7,
    '🎨 可视化中心': 8
}
```

## 依赖关系

### 复用组件

- `webapp/components/cusum_viz.py`：可视化组件（`plot_price_with_events()`, `render_sampling_rate_panel()`）
- `scripts/cusum_filtering.py`：`compute_dynamic_cusum_filter()` - 底层 CUSUM 采样实现

### 新增依赖

无

## 兼容性

### 特征工程页面 API 变更

**`webapp/utils/feature_calculator.py`**：
- `compute_all_features()` 新增参数 `cusum_data: Optional[pd.DataFrame]`
- 修改 `align_features_with_cusum()` 函数签名支持 DataFrame 输入
- 保留 `cusum_path` 参数作为备选（向后兼容）

**`webapp/pages/04_feature_engineering.py`**（原 03）：
- 优先从 `SessionManager.get('cusum_sampled_data')` 读取
- 若 Session 为空，fallback 到文件路径输入
- 移除 `cusum_path` 输入框（改用 Session 自动读取）

### 数据导入页面

移除 `bar_type='cusum'` 分支代码（第 310-418 行）。

## 验收标准

1. 新页面可独立访问，显示正确布局
2. 能从 Dollar Bars 执行 CUSUM 采样并显示结果
3. 能加载已保存的 CUSUM 结果文件
4. 采样结果可导出为 CSV/Parquet
5. 数据导入页面不再显示 CUSUM K 线选项
6. 侧边栏导航正确显示新页面
7. 采样结果可通过 Session 传递到特征工程页面
8. 无 Dollar Bars 数据时显示正确引导提示
9. 文件已存在时提示覆盖确认

## 实现优先级

1. 重命名现有页面文件（重新编号）
2. 创建新页面框架和基础布局
3. 实现数据源选择和参数配置
4. 实现执行采样和结果展示
5. 实现导出功能
6. 更新侧边栏导航（PAGES 字典、steps、step_mapping）
7. 清理数据导入页面的 CUSUM 代码
8. 更新特征工程页面兼容性
9. 端到端测试