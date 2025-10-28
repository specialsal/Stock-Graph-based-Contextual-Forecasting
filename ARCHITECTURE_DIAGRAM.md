# 架构图可视化

## 系统整体架构

```mermaid
graph TB
    subgraph "数据获取层"
        A1[聚宽数据源 RQData]
        A2[AKShare数据源]
        A3[交易日历数据]
        A4[股票基础信息]
    end
    
    subgraph "特征工程层"
        B1[技术指标特征]
        B2[量价特征]
        B3[波动特征]
        B4[动量特征]
    end
    
    subgraph "上下文特征层"
        C1[市场指数特征]
        C2[风格板块特征]
        C3[行业特征]
        C4[宏观环境特征]
    end
    
    subgraph "模型架构层"
        D1[DailyEncoder<br/>日频编码器]
        D2[DynamicGraphBlock<br/>图神经网络]
        D3[FiLM融合<br/>上下文门控]
        D4[Ranking Head<br/>排序输出层]
    end
    
    subgraph "训练框架层"
        E1[滚动训练策略]
        E2[Pairwise RankNet]
        E3[早停机制]
        E4[模型保存]
    end
    
    subgraph "回测框架层"
        F1[预测生成]
        F2[持仓优化]
        F3[绩效评估]
        F4[组合分析]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B1
    A4 --> B1
    
    B1 --> C1
    B2 --> C1
    B3 --> C1
    B4 --> C1
    
    C1 --> D1
    C2 --> D2
    C3 --> D2
    C4 --> D3
    
    D1 --> D2
    D2 --> D3
    D3 --> D4
    
    D4 --> E1
    E1 --> E2
    E2 --> E3
    E3 --> E4
    
    E4 --> F1
    F1 --> F2
    F2 --> F3
    F3 --> F4
    
    classDef dataLayer fill:#e1f5fe
    classDef featureLayer fill:#f3e5f5
    classDef contextLayer fill:#e8f5e8
    classDef modelLayer fill:#fff3e0
    classDef trainLayer fill:#ffebee
    classDef backtestLayer fill:#f1f8e9
    
    class A1,A2,A3,A4 dataLayer
    class B1,B2,B3,B4 featureLayer
    class C1,C2,C3,C4 contextLayer
    class D1,D2,D3,D4 modelLayer
    class E1,E2,E3,E4 trainLayer
    class F1,F2,F3,F4 backtestLayer
```

## 数据流架构

```mermaid
flowchart LR
    subgraph "数据源"
        S1[聚宽 RQData]
        S2[AKShare]
    end
    
    subgraph "数据处理"
        P1[数据获取<br/>data_acquire.py]
        P2[特征工程<br/>feature_engineering.py]
        P3[上下文特征<br/>feature_context.py]
        P4[标签生成<br/>label_generation.py]
    end
    
    subgraph "模型训练"
        T1[滚动训练<br/>train_rolling.py]
        T2[模型保存<br/>model.pth]
    end
    
    subgraph "回测评估"
        B1[预测生成<br/>backtest_rolling.py]
        B2[持仓优化<br/>btr_backtest.py]
        B3[绩效评估<br/>btr_metrics.py]
    end
    
    S1 --> P1
    S2 --> P1
    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> T1
    T1 --> T2
    T2 --> B1
    B1 --> B2
    B2 --> B3
    
    classDef source fill:#bbdefb
    classDef process fill:#c8e6c9
    classDef train fill:#ffecb3
    classDef backtest fill:#d7ccc8
    
    class S1,S2 source
    class P1,P2,P3,P4 process
    class T1,T2 train
    class B1,B2,B3 backtest
```

## 模块依赖关系

```mermaid
graph TD
    C[config.py<br/>全局配置] --> U[utils.py<br/>通用工具]
    
    U --> DA[data_acquire.py<br/>数据获取]
    U --> FE[feature_engineering.py<br/>特征工程]
    U --> FC[feature_context.py<br/>上下文特征]
    
    DA --> RP[run_pipeline.py<br/>主流程控制]
    FE --> RP
    FC --> RP
    
    RP --> TR[train_rolling.py<br/>滚动训练]
    RP --> LG[label_generation.py<br/>标签生成]
    
    TR --> MD[model.py<br/>模型定义]
    TR --> TU[train_utils.py<br/>训练工具]
    
    MD --> BR[backtest_rolling.py<br/>回测预测]
    TU --> BR
    
    BR --> BT[btr_backtest.py<br/>回测框架]
    BR --> BM[btr_metrics.py<br/>绩效评估]
    
    BT --> OL[optim_l2qp.py<br/>优化器]
    BT --> OP[optimize_position.py<br/>持仓优化]
    
    BM --> UB[utils_backtest.py<br/>回测工具]
    
    classDef config fill:#ffcdd2
    classDef utils fill:#f8bbd9
    classDef data fill:#e1bee7
    classDef feature fill:#d1c4e9
    classDef model fill:#c5cae9
    classDef train fill:#bbdefb
    classDef backtest fill:#b3e5fc
    classDef optimize fill:#b2ebf2
    
    class C config
    class U utils
    class DA,FC data
    class FE feature
    class MD model
    class TR,TU train
    class BR,BT,BM backtest
    class OL,OP optimize
```

## 模型组件架构

```mermaid
graph TB
    subgraph "输入层"
        I1[股票日频特征]
        I2[市场上下文特征]
        I3[行业图结构]
    end
    
    subgraph "特征编码器"
        E1[DailyEncoder<br/>CNN特征提取]
        E2[时间序列编码]
    end
    
    subgraph "图神经网络"
        G1[DynamicGraphBlock<br/>GAT注意力机制]
        G2[图卷积操作]
        G3[节点特征更新]
    end
    
    subgraph "特征融合"
        F1[FiLM融合<br/>特征线性调制]
        F2[上下文门控]
    end
    
    subgraph "输出层"
        O1[Ranking Head<br/>排序分数]
        O2[股票排名]
    end
    
    I1 --> E1
    I2 --> F1
    I3 --> G1
    
    E1 --> E2
    E2 --> G1
    
    G1 --> G2
    G2 --> G3
    G3 --> F1
    
    F1 --> F2
    F2 --> O1
    O1 --> O2
    
    classDef input fill:#ffebee
    classDef encoder fill:#f3e5f5
    classDef graph fill:#e8f5e8
    classDef fusion fill:#e1f5fe
    classDef output fill:#fff3e0
    
    class I1,I2,I3 input
    class E1,E2 encoder
    class G1,G2,G3 graph
    class F1,F2 fusion
    class O1,O2 output
```

## 训练流程架构

```mermaid
sequenceDiagram
    participant M as 主流程
    participant D as 数据加载
    participant T as 训练循环
    participant V as 验证评估
    participant S as 模型保存
    
    M->>D: 加载训练数据窗口
    D->>T: 准备批次数据
    
    loop 每个epoch
        T->>T: 前向传播
        T->>T: 计算Pairwise损失
        T->>T: 反向传播
        T->>T: 参数更新
        
        T->>V: 验证集评估
        V->>V: 计算RankIC
        V->>T: 返回验证结果
        
        alt 早停条件满足
            T->>S: 保存最佳模型
            T->>M: 结束训练
        else 继续训练
            T->>D: 下一个批次
        end
    end
    
    S->>M: 训练完成
```

## 文件组织结构

```
Stock-Graph-based-Contextual-Forecasting/
├── 数据层 (Data Layer)
│   ├── data_acquire.py          # 数据获取模块
│   ├── dataset.py               # 数据集定义
│   └── 数据存储文件/
│       ├── stock_price_day.parquet
│       ├── index_price_day.parquet
│       └── sector_price_day.parquet
├── 特征层 (Feature Layer)
│   ├── feature_engineering.py    # 特征工程
│   ├── feature_context.py       # 上下文特征
│   └── label_generation.py      # 标签生成
├── 模型层 (Model Layer)
│   ├── model.py                 # 模型定义
│   └── train_utils.py           # 训练工具
├── 训练层 (Training Layer)
│   ├── train_rolling.py         # 滚动训练
│   └── optim_l2qp.py            # 优化器
├── 回测层 (Backtest Layer)
│   ├── backtest_rolling.py      # 回测预测
│   ├── btr_backtest.py          # 回测框架
│   ├── btr_metrics.py           # 绩效评估
│   └── utils_backtest.py       # 回测工具
├── 工具层 (Utility Layer)
│   ├── utils.py                 # 通用工具
│   └── config.py                # 全局配置
└── 主流程 (Main Pipeline)
    ├── run_pipeline.py          # 主执行流程
    └── combine_s_g.py           # 组合分析
```

这个架构图展示了系统的层次化设计和模块间的依赖关系，帮助理解整个工程的运作机制。