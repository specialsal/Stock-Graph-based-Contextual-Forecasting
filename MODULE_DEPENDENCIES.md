# 模块依赖关系矩阵

## 文件依赖关系表

| 模块文件 | 依赖的文件 | 被依赖的文件 | 主要功能 |
|---------|-----------|-------------|----------|
| `config.py` | 无 | `utils.py`, `run_pipeline.py`, `data_acquire.py`, `feature_engineering.py`, `feature_context.py`, `label_generation.py`, `train_rolling.py`, `backtest_rolling.py` | 全局配置管理，定义所有参数和路径 |
| `utils.py` | `config.py` | `data_acquire.py`, `feature_engineering.py`, `feature_context.py`, `label_generation.py`, `train_rolling.py`, `backtest_rolling.py`, `utils_backtest.py` | 通用工具函数，数据加载、日历处理、指标计算 |
| `data_acquire.py` | `config.py`, `utils.py` | `run_pipeline.py` | 数据获取模块，从聚宽和AKShare获取原始数据 |
| `feature_engineering.py` | `config.py`, `utils.py` | `run_pipeline.py` | 特征工程，生成技术指标和量价特征 |
| `feature_context.py` | `config.py`, `utils.py` | `run_pipeline.py` | 上下文特征生成，市场指数和风格特征 |
| `label_generation.py` | `config.py`, `utils.py` | `run_pipeline.py` | 标签生成，基于未来收益率的排序标签 |
| `model.py` | 无 | `train_rolling.py`, `backtest_rolling.py` | 模型定义，GCFNet架构组件 |
| `train_utils.py` | `config.py`, `utils.py`, `model.py` | `train_rolling.py` | 训练辅助工具，Scaler拟合、数据加载 |
| `train_rolling.py` | `config.py`, `utils.py`, `model.py`, `train_utils.py` | `run_pipeline.py` | 滚动训练主循环，模型训练和验证 |
| `backtest_rolling.py` | `config.py`, `utils.py`, `model.py`, `train_utils.py` | `run_pipeline.py`, `btr_backtest.py` | 回测预测生成，模型推理 |
| `btr_backtest.py` | `config.py`, `utils.py`, `optim_l2qp.py`, `optimize_position.py` | `run_pipeline.py`, `btr_metrics.py` | 回测框架，持仓优化和组合构建 |
| `btr_metrics.py` | `config.py`, `utils.py`, `utils_backtest.py` | `run_pipeline.py` | 绩效评估，计算各种风险收益指标 |
| `utils_backtest.py` | `config.py`, `utils.py` | `btr_metrics.py` | 回测工具函数，最大回撤、年化收益等计算 |
| `optim_l2qp.py` | 无 | `btr_backtest.py` | L2正则化二次规划优化器 |
| `optimize_position.py` | 无 | `btr_backtest.py` | 持仓优化算法 |
| `run_pipeline.py` | `config.py`, `data_acquire.py`, `feature_engineering.py`, `feature_context.py`, `label_generation.py`, `train_rolling.py`, `backtest_rolling.py`, `btr_backtest.py`, `btr_metrics.py` | 无 | 主流程控制，协调所有模块执行 |

## 调用关系图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           核心依赖关系图                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  config.py (配置中心)                                                    │
│       ↓                                                                 │
│  utils.py (工具库)                                                       │
│       ↓                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                   数据处理模块                                  │    │
│  │  data_acquire.py → feature_engineering.py → feature_context.py   │    │
│  │                    ↓                                            │    │
│  │              label_generation.py                               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       ↓                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                   模型训练模块                                  │    │
│  │  model.py ← train_utils.py ← train_rolling.py                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       ↓                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                   回测评估模块                                  │    │
│  │  backtest_rolling.py → btr_backtest.py → btr_metrics.py        │    │
│  │                    ↓              ↓                            │    │
│  │              optim_l2qp.py   optimize_position.py              │    │
│  │                    ↓              ↓                            │
│  │              utils_backtest.py                                │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       ↓                                                                 │
│  run_pipeline.py (主流程协调)                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 模块功能详细说明

### 1. 配置管理模块 (`config.py`)
**核心依赖**: 无
**被依赖模块**: 所有其他模块
**功能**:
- 定义数据路径、模型参数、训练配置
- 全局参数管理，确保一致性
- 支持不同环境的配置切换

### 2. 工具库模块 (`utils.py`)
**核心依赖**: `config.py`
**被依赖模块**: 数据处理、特征工程、训练、回测模块
**功能**:
- 通用数据加载和预处理
- 交易日历处理
- 技术指标计算
- 文件操作工具

### 3. 数据处理模块组

#### 3.1 数据获取 (`data_acquire.py`)
**核心依赖**: `config.py`, `utils.py`
**被依赖模块**: `run_pipeline.py`
**功能**:
- 从聚宽RQData获取股票数据
- 从AKShare获取补充数据
- 增量更新机制
- 数据质量检查

#### 3.2 特征工程 (`feature_engineering.py`)
**核心依赖**: `config.py`, `utils.py`
**被依赖模块**: `run_pipeline.py`
**功能**:
- 技术指标计算（RSI、MACD等）
- 量价特征生成
- 多线程并行处理
- 增量特征计算

#### 3.3 上下文特征 (`feature_context.py`)
**核心依赖**: `config.py`, `utils.py`
**被依赖模块**: `run_pipeline.py`
**功能**:
- 市场指数特征提取
- 风格板块特征生成
- 行业轮动特征
- 宏观环境指标

#### 3.4 标签生成 (`label_generation.py`)
**核心依赖**: `config.py`, `utils.py`
**被依赖模块**: `run_pipeline.py`
**功能**:
- 基于未来收益率的排序标签
- 多空标签生成
- 标签质量控制

### 4. 模型训练模块组

#### 4.1 模型定义 (`model.py`)
**核心依赖**: 无
**被依赖模块**: `train_rolling.py`, `backtest_rolling.py`
**功能**:
- GCFNet模型架构定义
- DailyEncoder、DynamicGraphBlock、FiLM组件
- 前向传播逻辑

#### 4.2 训练工具 (`train_utils.py`)
**核心依赖**: `config.py`, `utils.py`, `model.py`
**被依赖模块**: `train_rolling.py`
**功能**:
- Scaler拟合和转换
- RAM模式数据加载
- 训练辅助函数

#### 4.3 滚动训练 (`train_rolling.py`)
**核心依赖**: `config.py`, `utils.py`, `model.py`, `train_utils.py`
**被依赖模块**: `run_pipeline.py`
**功能**:
- 滚动训练主循环
- Pairwise RankNet损失函数
- 早停机制
- 模型保存

### 5. 回测评估模块组

#### 5.1 回测预测 (`backtest_rolling.py`)
**核心依赖**: `config.py`, `utils.py`, `model.py`, `train_utils.py`
**被依赖模块**: `run_pipeline.py`, `btr_backtest.py`
**功能**:
- 模型推理和预测生成
- 特征标准化
- 预测结果保存

#### 5.2 回测框架 (`btr_backtest.py`)
**核心依赖**: `config.py`, `utils.py`, `optim_l2qp.py`, `optimize_position.py`
**被依赖模块**: `run_pipeline.py`, `btr_metrics.py`
**功能**:
- 持仓优化
- 组合构建
- 交易成本计算

#### 5.3 绩效评估 (`btr_metrics.py`)
**核心依赖**: `config.py`, `utils.py`, `utils_backtest.py`
**被依赖模块**: `run_pipeline.py`
**功能**:
- 风险收益指标计算
- RankIC分析
- 绩效报告生成

#### 5.4 优化器模块 (`optim_l2qp.py`, `optimize_position.py`)
**核心依赖**: 无
**被依赖模块**: `btr_backtest.py`
**功能**:
- L2正则化二次规划
- 持仓优化算法

#### 5.5 回测工具 (`utils_backtest.py`)
**核心依赖**: `config.py`, `utils.py`
**被依赖模块**: `btr_metrics.py`
**功能**:
- 最大回撤计算
- 年化收益率计算
- 夏普比率计算

### 6. 主流程控制 (`run_pipeline.py`)
**核心依赖**: 所有其他模块
**被依赖模块**: 无
**功能**:
- 协调整个系统执行流程
- 模块调用顺序控制
- 错误处理和日志记录

## 数据流依赖关系

```
外部数据源
    ↓
data_acquire.py (原始数据获取)
    ↓
feature_engineering.py (技术特征)
    ↓
feature_context.py (上下文特征)
    ↓
label_generation.py (标签生成)
    ↓
train_rolling.py (模型训练)
    ↓
backtest_rolling.py (预测生成)
    ↓
btr_backtest.py (持仓优化)
    ↓
btr_metrics.py (绩效评估)
    ↓
最终结果输出
```

## 关键依赖路径

### 训练路径
```
config.py → utils.py → data_acquire.py → feature_engineering.py → 
feature_context.py → label_generation.py → model.py → train_utils.py → 
train_rolling.py
```

### 回测路径
```
config.py → utils.py → model.py → train_utils.py → backtest_rolling.py → 
btr_backtest.py → btr_metrics.py → utils_backtest.py
```

### 优化路径
```
optim_l2qp.py → btr_backtest.py
optimize_position.py → btr_backtest.py
```

这个依赖关系矩阵清晰地展示了整个工程中各个模块之间的调用关系，有助于理解系统的架构设计和模块间的协作方式。