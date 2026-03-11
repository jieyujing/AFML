# CSV 数据支持与 Dollar Bar 生成设计文档

**日期**: 2026-03-11
**状态**: 已批准
**分支**: feature/research-webui

---

## 1. 概述

### 1.1 目标

为 AFMLKit Web UI 添加：
1. CSV 格式数据加载支持（读取现有文件 + 上传新文件）
2. 1 分钟 Dollar Bar 生成功能（基于每日目标参数）
3. IID 评估与可视化

### 1.2 背景

当前 Web UI 已具备基础架构，但需要：
- 支持从 `data/csv` 目录加载 1 分钟 OHLCV 数据
- 实现动态 Dollar Bar 生成（参考 `scripts/evaluate_dollar_bars.py`）
- 提供交互式参数配置和结果可视化

---

## 2. 需求规格

### 2.1 功能需求

| ID | 需求 | 优先级 |
|----|------|--------|
| F1 | 从 `data/csv` 目录选择并加载 CSV 文件 | P0 |
| F2 | 支持上传外部 CSV 文件 | P0 |
| F3 | 自动识别和验证 OHLCV 列 | P0 |
| F4 | 配置每日目标 Dollar Bar 参数（默认 [4, 6, 10, 20, 50]） | P0 |
| F5 | 流式生成 Dollar Bars（多频率并行） | P0 |
| F6 | 导出各频率 CSV 文件 | P0 |
| F7 | 实时预览生成进度和统计 | P1 |
| F8 | IID 评估（JB 检验、一阶自相关） | P1 |
| F9 | 可视化（收益分布、对比分析） | P1 |

### 2.2 非功能需求

- **性能**: 支持 25GB+ 数据流式处理，不一次性加载到内存
- **可用性**: 进度条实时更新，用户可随时查看状态
- **兼容性**: 与现有 Web UI 架构一致

---

## 3. 架构设计

### 3.1 组件架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Web UI                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  CSV 加载组件     │  │ Dollar Bar 配置  │  │ 结果展示    │ │
│  │  - 目录选择      │  │  - 目标频率      │  │  - 统计     │ │
│  │  - 文件上传      │  │  - EWMA Span    │  │  - 图表     │ │
│  │  - 列映射        │  │  - 运行按钮      │  │  - 导出     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    核心处理层                                 │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  CSV 加载器       │  │ Dollar Bar 生成器 │  │ IID 评估器   │ │
│  │  - 格式验证      │  │  - 阶段 1 扫描      │  │  - JB 检验   │ │
│  │  - 数据解析      │  │  - 阶段 2 生成      │  │  - 自相关   │ │
│  │  - 错误处理      │  │  - 流式输出      │  │  - 评分     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      数据持久层                              │
├─────────────────────────────────────────────────────────────┤
│  - outputs/dollar_bars/  (生成的 Dollar Bar CSV)            │
│  - outputs/visualizations/ (评估图表)                        │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 数据流

```
用户操作 → CSV 加载 → 数据验证 → 计算日度 dollar volume →
EWMA 阈值计算 → 动态 Dollar Bar 生成 →
├─→ CSV 导出
├─→ IID 评估
└─→ 可视化生成
```

---

## 4. 模块设计

### 4.1 CSV 加载器 (`webapp/utils/csv_loader.py`)

```python
class CSVDataloader:
    """CSV 数据加载器"""

    def __init__(self, data_dir: str = "data/csv"):
        self.data_dir = Path(data_dir)

    def list_available_files(self) -> List[Path]:
        """列出目录下所有 CSV 文件"""

    def load_file(self, filepath: str, column_mapping: Dict = None) -> pd.DataFrame:
        """加载 CSV 文件并验证格式"""

    def validate_ohlcv(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """验证 OHLCV 格式"""
```

### 4.2 Dollar Bar 生成器 (`webapp/utils/dollar_bar_generator.py`)

```python
class DollarBarGenerator:
    """Dollar Bar 流式生成器"""

    def __init__(self,
                 target_daily_bars: List[int] = [4, 6, 10, 20, 50],
                 ewma_span: int = 20):
        self.target_daily_bars = target_daily_bars
        self.ewma_span = ewma_span

    def scan_daily_dollar_volume(self, df: pd.DataFrame) -> pd.Series:
        """阶段 1: 计算每日 dollar volume"""

    def compute_ewma_thresholds(self, daily_vol: pd.Series) -> Dict[int, pd.Series]:
        """计算 EWMA 阈值"""

    def generate_bars_streaming(self,
                                df: pd.DataFrame,
                                thresholds: Dict[int, pd.Series]) -> Dict[int, pd.DataFrame]:
        """阶段 2: 流式生成 Dollar Bars"""

    def save_bars(self, bars_dict: Dict[int, pd.DataFrame], output_dir: str):
        """保存各频率 CSV"""
```

### 4.3 IID 评估器 (`webapp/utils/iid_evaluator.py`)

```python
class IIDEvaluator:
    """IID 评估器"""

    def evaluate(self, ohlcv: pd.DataFrame) -> Dict:
        """执行 IID 评估"""

    def jarque_bera_test(self, returns: pd.Series) -> Tuple[float, float]:
        """JB 检验"""

    def autocorrelation(self, returns: pd.Series, lag: int = 1) -> float:
        """计算自相关"""

    def score(self, results: Dict[int, Dict]) -> int:
        """综合评分，返回最优频率"""
```

### 4.4 可视化组件 (`webapp/components/dollar_bar_viz.py`)

```python
def plot_iid_comparison(results: Dict[int, Dict], best_freq: int) -> go.Figure:
    """IID 指标对比图"""

def plot_return_distribution(returns: pd.Series, best_freq: int) -> go.Figure:
    """收益分布图"""

def plot_bars_count_comparison(bars_dict: Dict[int, pd.DataFrame]) -> go.Figure:
    """各频率 bar 数量对比"""

def plot_time_series_comparison(bars_dict: Dict[int, pd.DataFrame]) -> go.Figure:
    """时间序列对比"""
```

---

## 5. Web UI 页面设计

### 5.1 数据导入页面增强 (`pages/01_data_import.py`)

在现有页面中添加：
- **CSV 数据源选项卡**: 显示 `data/csv` 目录下的文件列表
- **列自动识别**: 映射 `datetime` → 索引，`close` → 价格等
- **1 分钟 K 线到 Dollar Bar 转换入口**

### 5.2 新增 Dollar Bar 生成页面 (`pages/02_dollar_bar_generator.py`)

```python
# 页面结构
st.title("💵 Dollar Bar 生成")

# 步骤 1: 数据选择
selected_file = st.selectbox("选择数据文件", available_files)

# 步骤 2: 参数配置
st.markdown("### 参数配置")
target_bars = st.multiselect("每日目标 Bars 数", [4, 6, 10, 20, 50], default=[4, 6, 10, 20, 50])
ewma_span = st.slider("EWMA Span", 5, 50, 20)

# 步骤 3: 运行生成
if st.button("开始生成"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 调用生成器
    generator = DollarBarGenerator(target_daily_bars=target_bars, ewma_span=ewma_span)
    results = generator.run(data_df, progress_callback=progress_bar.progress)

    # 显示结果
    st.success("生成完成!")

    # 显示 IID 评估
    evaluator = IIDEvaluator()
    eval_results = {freq: evaluator.evaluate(df) for freq, df in results.items()}
    best_freq = evaluator.score(eval_results)
    st.info(f"最优频率：{best_freq} bars/day")

    # 可视化
    st.plotly_chart(plot_iid_comparison(eval_results, best_freq))

# 步骤 4: 导出
for freq, df in results.items():
    csv = df.to_csv()
    st.download_button(f"下载频率 {freq} CSV", csv, f"dollar_bars_{freq}.csv")
```

---

## 6. 文件结构

```
webapp/
├── utils/
│   ├── csv_loader.py           # 新增：CSV 加载器
│   ├── dollar_bar_generator.py # 新增：Dollar Bar 生成器
│   └── iid_evaluator.py        # 新增：IID 评估器
├── components/
│   └── dollar_bar_viz.py       # 新增：Dollar Bar 可视化
└── pages/
    ├── 01_data_import.py       # 修改：添加 CSV 支持
    └── 02_dollar_bar_generator.py # 新增：Dollar Bar 生成页面
outputs/
├── dollar_bars/                # 生成的 Dollar Bar CSV
└── visualizations/             # 评估图表
```

---

## 7. 错误处理

| 错误类型 | 处理方式 |
|---------|---------|
| CSV 格式错误 | 显示详细错误信息，提示缺少的列 |
| 内存不足 | 流式处理，分批加载，自动 GC |
| 生成失败 | 保留已生成部分，提供错误日志下载 |
| 数据不连续 | 警告提示，自动填充缺失时间段 |

---

## 8. 测试计划

1. **单元测试**: 测试 CSV 加载器、Dollar Bar 生成器、IID 评估器
2. **集成测试**: 使用 `data/csv` 目录下的真实数据测试完整流程
3. **性能测试**: 测试大文件（>1GB）处理性能
4. **UI 测试**: 验证所有交互功能正常

---

## 9. 验收标准

- [ ] 可以从 `data/csv` 目录选择并加载文件
- [ ] 可以上传外部 CSV 文件
- [ ] 正确生成 5 种频率的 Dollar Bars
- [ ] 导出的 CSV 包含正确的 OHLCV 数据
- [ ] IID 评估结果与脚本版本一致
- [ ] 可视化图表正确渲染
- [ ] 进度条实时更新

---

## 10. 参考资料

- `scripts/evaluate_dollar_bars.py` - 原始评估脚本
- `afmlkit/bar/kit.py` - Dollar Bar Kit 实现
- `afmlkit/bar/logic.py` - Dollar Bar 核心逻辑
