# CUSUM 采样页面设计规范

## 概述

将 CUSUM Filter 功能从"数据导入"页面拆分为独立页面，放置在 Dollar Bar 生成之后、特征工程之前，形成更清晰的数据处理流程。

## 变更范围

### 新增

- 新页面：`webapp/pages/02a_cusum_sampling.py`（CUSUM 采样）

### 修改

- `webapp/pages/01_data_import.py`：移除 CUSUM K 线选项
- `webapp/components/sidebar.py`：更新导航菜单，添加新页面
- `webapp/session.py`：添加 CUSUM 相关状态键

### 删除

- 数据导入页面中 `bar_type='cusum'` 相关代码（约 80 行）

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

`webapp/pages/02a_cusum_sampling.py`

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
- 输出：`cusum_source_data` (Session)

#### 2. 参数配置区块

**功能**：配置 CUSUM 采样参数

**UI 元素**：
- 高级模式 Checkbox（默认关闭）
- `vol_span`：波动率窗口（默认 50，高级模式可调）
- `threshold_multiplier`：阈值乘数（默认 2.0，高级模式可调）
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
- 文件名输入框
- 导出按钮

**导出路径**：`outputs/cusum_sampling/`

## Session 状态键

新增以下状态键到 `SessionManager.KEYS`：

```python
'cusum_source_data',    # 输入数据
'cusum_config',         # 参数配置
'cusum_sampled_data',   # 采样后数据
'cusum_events',         # CUSUM 事件时间点数组
```

## 侧边栏导航更新

### PAGES 字典更新

```python
PAGES = {
    "首页": {...},
    "1️⃣ 数据导入": {...},
    "💵 Dollar Bar": {...},
    "🔬 CUSUM 采样": {  # 新增
        "icon": "🔬",
        "file": "pages/02a_cusum_sampling.py",
        "description": "CUSUM 事件采样与可视化"
    },
    "2️⃣ 特征工程": {...},  # 编号调整为 2️⃣
    ...
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

## 依赖关系

### 复用组件

- `webapp/components/cusum_viz.py`：可视化组件
- `webapp/utils/feature_calculator.py`：CUSUM 计算逻辑（需确认是否可独立调用）
- `scripts/cusum_filtering.py`：底层 CUSUM 实现

### 新增依赖

无

## 兼容性

### 特征工程页面

移除 `cusum_path` 参数配置，改为直接从 `SessionManager.get('cusum_sampled_data')` 读取。

### 数据导入页面

移除 `bar_type='cusum'` 分支代码（约 80 行）。

## 验收标准

1. 新页面可独立访问，显示正确布局
2. 能从 Dollar Bars 执行 CUSUM 采样并显示结果
3. 能加载已保存的 CUSUM 结果文件
4. 采样结果可导出为 CSV/Parquet
5. 数据导入页面不再显示 CUSUM K 线选项
6. 侧边栏导航正确显示新页面
7. 采样结果可通过 Session 传递到特征工程页面

## 实现优先级

1. 创建新页面框架和基础布局
2. 实现数据源选择和参数配置
3. 实现执行采样和结果展示
4. 实现导出功能
5. 更新侧边栏导航
6. 清理数据导入页面的 CUSUM 代码
7. 端到端测试