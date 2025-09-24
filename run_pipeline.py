# coding: utf-8
"""
一键执行：特征 ➜ 上下文特征 ➜ 标签 ➜ 滚动训练
"""
from data_acquire import main as acquire_data
from feature_engineering import main as gen_feature
from feature_context import main as gen_context
from label_generation import main as gen_label
from train_rolling import main as rolling_train
from backtest_rolling import main as bt_gen
from btr_backtest import main as position_gen
from btr_metrics import main as metrics_gen
from backtest_rolling_config import BTRollingConfig
from combine_s_g import main as combine_s_g

def main():
    print("STEP 0: 更新数据")
    acquire_data()
    print("STEP 1: 更新离线特征仓")
    gen_feature()
    print("STEP 2: 更新市场/风格上下文特征")
    gen_context()
    print("STEP 3: 更新周度标签")
    gen_label()
    print("STEP 4: 滚动训练")
    rolling_train()
    # print("STEP 5: 滚动生成打分")
    # bt_gen()
    # print("STEP 6: 滚动回测")
    # position_gen()
    # print("STEP 7: 回测统计指标")
    # metrics_gen(f"./backtest_rolling/{BTRollingConfig.run_name}/nav_{BTRollingConfig.run_name}.csv")
    # print("STEP 8: 组合黄金etf")
    # combine_s_g()
    # print("STEP 9: 组合回测统计指标")
    # metrics_gen(f"./backtest_rolling/{BTRollingConfig.run_name}/s_g_combo/combo_nav.csv")


if __name__ == "__main__":
    main()