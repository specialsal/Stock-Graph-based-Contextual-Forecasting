# coding: utf-8
"""
一键执行：特征 ➜ 上下文特征 ➜ 标签 ➜ 滚动训练
"""
from feature_engineering import main as gen_feature
from feature_context import main as gen_context
from label_generation import main as gen_label
from train_rolling import main as rolling_train

def main():
    print("STEP 1: 生成离线特征仓")
    gen_feature()
    print("STEP 2: 生成市场/风格上下文特征")
    gen_context()
    print("STEP 3: 生成周度标签")
    gen_label()
    print("STEP 4: 滚动训练")
    rolling_train()

if __name__ == "__main__":
    main()