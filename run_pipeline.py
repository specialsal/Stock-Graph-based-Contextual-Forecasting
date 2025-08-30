# coding: utf-8
"""
运行整个pipeline
"""
import sys
from pathlib import Path


def main():
    print("=" * 60)
    print("量化模型训练Pipeline")
    print("=" * 60)

    # 1. 生成标签
    print("\n步骤1: 生成标签...")
    from label_generation import main as generate_labels
    generate_labels()

    # 2. 数据预处理
    print("\n步骤2: 数据预处理...")
    from data_preprocessing import main as preprocess_data
    preprocess_data()

    # 3. 模型训练
    print("\n步骤3: 模型训练...")
    from train import main as train_model
    train_model()

    print("\n训练完成！")


if __name__ == "__main__":
    main()