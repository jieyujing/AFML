# AFMLKit Web UI 使用指南

## 快速开始

### 安装依赖

```bash
# 安装 webapp 依赖
pip install -e ".[webapp]"

# 或者手动安装
pip install streamlit>=1.30.0 plotly>=5.18.0 pyyaml>=6.0 statsmodels>=0.14.0
```

### 启动应用

```bash
# 方法 1: 直接运行
streamlit run webapp/app.py

# 方法 2: 使用模块运行
python -m webapp

# 方法 3: 使用运行脚本
python run_webapp.py
```

应用将在 `http://localhost:8501` 启动。

## 功能模块

### 1. 数据导入 (📥)

**功能**:
- 从 `data/csv` 目录加载现有 CSV 文件
- 上传 CSV/Parquet/HDF5 格式的交易数据
- 支持生成示例数据用于测试
- 自动识别 OHLCV 列
- K 线构建 (时间/Tick/成交量/金额/CUSUM)

**操作流程**:
1. 选择数据源 (从 data/csv 目录/上传文件/示例数据)
2. 确认列映射
3. 验证数据完整性
4. 选择 K 线类型并构建

### 💵 Dollar Bar 生成 (新增)

**功能**:
- 从 `data/csv` 目录加载 1 分钟 OHLCV 数据
- 动态阈值 Dollar Bar 生成（多频率并行）
- IID 评估（Jarque-Bera 检验、自相关分析）
- 最优频率推荐
- 交互式可视化

**操作流程**:
1. 从 data/csv 目录选择文件
2. 配置每日目标 Bar 数（默认 [4, 6, 10, 20, 50]）
3. 运行生成
4. 查看 IID 评估结果和可视化
5. 导出 CSV

### 2. 特征工程 (🔧)

**功能**:
- 波动率特征 (EWM, Parkinson, ATR)
- 动量特征 (RSI, ROC, MACD)
- 分数阶差分 (FFD)
- 特征管道预览

**操作流程**:
1. 选择特征类别
2. 配置参数 (窗口大小等)
3. 执行特征计算
4. 预览和导出特征

### 3. 标签生成 (🏷️)

**功能**:
- 三重屏障法 (TBM) 标签生成
- 样本权重计算 (并发权重、收益归因、时间衰减)
- 标签分布可视化

**操作流程**:
1. 配置 TBM 参数 (止盈/止损/时间期限)
2. 生成标签
3. 计算样本权重
4. 分析标签分布

### 4. 特征分析 (📊)

**功能**:
- 特征聚类分析
- 特征相关性热力图
- Clustered MDA 特征重要性
- 特征选择

**操作流程**:
1. 执行特征聚类
2. 查看相关性矩阵
3. 计算特征重要性
4. 选择特征子集

### 5. 模型训练 (🤖)

**功能**:
- 支持随机森林/梯度提升/XGBoost
- Purged CV 交叉验证
- 超参数配置
- 模型评估指标

**操作流程**:
1. 选择模型类型
2. 配置超参数
3. 执行交叉验证
4. 训练模型并评估

### 6. 回测评估 (📈)

**功能**:
- 策略回测
- 绩效指标 (Sharpe, PSR, DSR, 最大回撤)
- 权益曲线可视化
- DSR 决策规则

**操作流程**:
1. 配置交易参数 (资金/手续费/滑点)
2. 运行回测
3. 查看绩效指标
4. 分析可视化结果

### 7. 可视化中心 (🎨)

**功能**:
- CUSUM 过滤事件图
- TBM 标签分布图
- 样本并发性和唯一性图
- 所有生成的可视化汇总

### 8. 实验管理 (📁)

**功能**:
- 保存实验配置和结果
- 加载历史实验
- 删除实验
- 实验对比

## 项目结构

```
webapp/
├── app.py                 # Streamlit 主入口
├── config.py              # 配置管理
├── session.py             # 会话状态管理
├── __main__.py            # 运行入口
├── components/            # 可复用 UI 组件
│   ├── __init__.py
│   ├── sidebar.py         # 侧边栏导航
│   ├── data_loader.py     # 数据上传/加载组件
│   ├── param_editor.py    # 参数编辑器
│   └── charts.py          # 通用图表组件
├── pages/                 # 各功能页面
│   ├── 01_data_import.py
│   ├── 03_feature_engineering.py
│   ├── 04_labeling.py
│   ├── 05_feature_analysis.py
│   ├── 06_model_training.py
│   ├── 07_backtest.py
│   ├── 08_visualization.py
│   └── 09_experiment.py
├── utils/                 # 工具函数
│   ├── __init__.py
│   ├── data_loader.py     # 数据加载器
│   ├── config_io.py       # 配置读写
│   └── export.py          # 导出功能
└── experiments/           # 实验存储目录 (动态生成)
```

## 会话状态管理

Web UI 使用 Streamlit 的 `session_state` 来管理跨页面的数据流：

```python
from session import SessionManager

# 初始化
SessionManager.init_session()

# 获取数据
bar_data = SessionManager.get('bar_data')

# 更新数据
SessionManager.update('features', features_df)

# 重置所有
SessionManager.reset_all()
```

## 数据流

```
数据导入 → K 线构建 → 特征工程 → 标签生成 → 特征分析 → 模型训练 → 回测评估
                ↓                                              ↓
           可视化中心 ←───────────────────────────────────────┘
```

## 常见问题

### Q: 如何保存我的研究进度？

使用"实验管理"页面保存当前配置和结果。加载实验时可以恢复之前的研究状态。

### Q: 数据量太大导致页面卡顿怎么办？

- 使用采样数据进行探索性分析
- 在特征工程页面使用渐进式加载
- 增加 Streamlit 的内存限制

### Q: 如何在后台运行长时间任务？

Streamlit 本身不适合长时间运行任务。对于批量训练，建议使用脚本方式运行。

### Q: 如何部署到服务器？

```bash
# 使用 nohup
nohup streamlit run webapp/app.py --server.port 8501 &

# 或使用 systemd 服务
```

## 扩展开发

### 添加新页面

1. 在 `pages/` 目录创建新文件，命名格式 `XX_page_name.py`
2. 在 `components/sidebar.py` 的 `PAGES` 字典中注册页面
3. 实现页面逻辑

### 添加新组件

1. 在 `components/` 目录创建新组件文件
2. 导出组件函数
3. 在页面中导入使用

### 添加新特征

1. 在 `afmlkit/feature/` 目录实现特征函数
2. 在 `pages/03_feature_engineering.py` 中调用

## 技术栈

- **前端框架**: Streamlit
- **图表库**: Plotly, Matplotlib
- **数据处理**: Pandas, NumPy
- **机器学习**: Scikit-learn
- **配置管理**: PyYAML

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

MIT License
