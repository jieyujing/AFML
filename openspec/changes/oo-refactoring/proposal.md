## Why

当前 AFML 项目采用过程式编程风格,各模块以独立函数形式实现,导致:
- **状态管理混乱**: 数据在各函数间传递,缺乏统一的上下文管理
- **可测试性差**: 难以对独立功能进行单元测试
- **可维护性低**: 修改一处可能影响多处,缺乏清晰的依赖关系
- **复用困难**: 核心逻辑与具体实现耦合,难以在不同场景复用

进行面向对象重构可以将核心金融 ML 算法封装为可组合的类,提高代码的模块化程度和可维护性。

## What Changes

将所有核心模块从过程式函数重构为面向对象类:

- **DollarBarsProcessor**: 美元Bars生成器类
- **TripleBarrierLabeler**: 三重障碍标签生成器类
- **FeatureEngineer**: 特征工程类
- **SampleWeightCalculator**: 样本权重计算器类
- **PurgedKFoldCV**: 已有(保持兼容)
- **MetaLabelingPipeline**: 元标签管道类
- **BetSizer**: 仓位管理类

## Capabilities

### New Capabilities
- `dollar-bars-processor`: 美元Bars生成器类,封装固定/动态阈值逻辑
- `triple-barrier-labeler`: 三重障碍标签生成器类,封装止盈/止损/时间障碍逻辑
- `feature-engineer`: 特征工程类,整合Alpha158和FFD特征
- `sample-weight-calculator`: 样本权重计算器类
- `meta-labeling-pipeline`: 元标签管道类
- `bet-sizer`: 仓位管理类

### Modified Capabilities
- 无(这是全新重构,不影响现有功能规格)

## Impact

- **src/process_bars.py** → 重构为 DollarBarsProcessor 类
- **src/labeling.py** → 重构为 TripleBarrierLabeler 类
- **src/features.py** → 重构为 FeatureEngineer 类
- **src/sample_weights.py** → 重构为 SampleWeightCalculator 类
- **src/meta_labeling.py** → 重构为 MetaLabelingPipeline 类
- **src/bet_sizing.py** → 重构为 BetSizer 类
- **src/cv_setup.py** → 保持 PurgedKFold 类(已有),添加管道集成
