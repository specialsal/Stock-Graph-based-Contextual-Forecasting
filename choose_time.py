import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ===================== 基本参数 =====================
run_name = "TGF-model"   # 替换成你的 run_name
base_dir = Path(f"./backtest_rolling/{run_name}")

# 进出场成本（虚拟 ETF 视角）
buy_cost_total = 0.001   # 0.1% 买入成本
sell_cost_total = 0.001  # 0.1% 卖出成本

# 滚动窗口参数
train_window = 252   # 每次训练用过去多少个交易日（可根据数据长度调整）
refit_step   = 21    # 每隔多少天重训一次模型（1 = 每天重训）

# ===================== 1. 读取四个文件 =====================
nav_path        = base_dir / f"nav_{run_name}.csv"
stock_ret_path  = base_dir / f"stock_returns_{run_name}.csv"
trades_path     = base_dir / f"stock_trades_{run_name}.csv"
positions_path  = base_dir / f"positions_{run_name}_equal.csv"

nav_df = pd.read_csv(nav_path, parse_dates=["date"])
nav_df = nav_df.sort_values("date").reset_index(drop=True)

stock_ret_df = pd.read_csv(stock_ret_path, parse_dates=["date"])
stock_ret_df = stock_ret_df.sort_values(["date", "stock"]).reset_index(drop=True)

trades_df = pd.read_csv(trades_path, parse_dates=["date"])
trades_df = trades_df.sort_values(["date", "stock"]).reset_index(drop=True)

pos_df = pd.read_csv(positions_path, parse_dates=["date"])
pos_df = pos_df.sort_values(["date", "stock"]).reset_index(drop=True)

# ===================== 2. 构造日度特征 =====================
dates = nav_df["date"]

# --- 2.1 从 stock_returns 聚合横截面特征 ---
g_ret = stock_ret_df.groupby("date")

def build_cross_section_features(g):
    df = g
    n = len(df)
    if n == 0:
        return pd.Series(dtype=float)

    w = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    r = pd.to_numeric(df["ret"], errors="coerce").fillna(0.0)

    n_up = (r > 0).sum()
    n_dn = (r <= 0).sum()
    ratio_up = n_up / n if n > 0 else 0.0

    w_up = w[r > 0].sum()
    w_dn = w[r <= 0].sum()
    w_sum = w_up + w_dn
    ratio_w_up = w_up / w_sum if w_sum > 0 else 0.0

    mean_ret = r.mean()
    wmean_ret = (w * r).sum()
    median_ret = r.median()
    std_ret = r.std(ddof=0)

    p10 = r.quantile(0.1)
    p50 = median_ret
    p90 = r.quantile(0.9)

    frac_big_up = (r > 0.02).mean()
    frac_big_dn = (r < -0.02).mean()

    return pd.Series({
        "cs_n": n,
        "cs_n_up": n_up,
        "cs_n_dn": n_dn,
        "cs_ratio_up": ratio_up,
        "cs_ratio_w_up": ratio_w_up,
        "cs_mean_ret": mean_ret,
        "cs_wmean_ret": wmean_ret,
        "cs_median_ret": median_ret,
        "cs_std_ret": std_ret,
        "cs_p10_ret": p10,
        "cs_p50_ret": p50,
        "cs_p90_ret": p90,
        "cs_frac_big_up": frac_big_up,
        "cs_frac_big_dn": frac_big_dn,
    })

cs_feat = g_ret.apply(build_cross_section_features)
cs_feat.index.name = "date"
cs_feat = cs_feat.reset_index()

# --- 2.2 从 positions 聚合“持仓结构”特征 ---
g_pos = pos_df.groupby("date")

def build_position_features(g):
    df = g
    n_pos = df["stock"].nunique()
    sc = pd.to_numeric(df.get("score", pd.Series(index=df.index, data=np.nan)), errors="coerce")
    mean_score = sc.mean()
    std_score = sc.std(ddof=0)
    median_score = sc.median()

    return pd.Series({
        "pos_n_stock": n_pos,
        "pos_mean_score": mean_score,
        "pos_std_score": std_score,
        "pos_median_score": median_score,
    })

pos_feat = g_pos.apply(build_position_features)
pos_feat.index.name = "date"
pos_feat = pos_feat.reset_index()

# --- 2.3 从 trades 聚合“交易强度”特征 ---
if not trades_df.empty:
    g_trd = trades_df.groupby("date")

    def build_trade_features(g):
        df = g
        n_trades = len(df)
        n_buy = (df["action"] == "BUY").sum()
        n_sell = (df["action"] == "SELL").sum()
        n_tp = (df["action"] == "TP").sum()
        n_sl = (df["action"] == "SL").sum()

        n_stop = n_tp + n_sl
        buy_sell_ratio = n_buy / max(1, n_sell)
        stop_ratio = n_stop / max(1, n_trades)

        dir_map = {"BUY": 1, "SELL": -1, "TP": -1, "SL": -1}
        net_dir = df["action"].map(dir_map).fillna(0).sum()
        net_dir_norm = net_dir / max(1, n_trades)

        return pd.Series({
            # "trd_n": n_trades,
            # "trd_n_buy": n_buy,
            # "trd_n_sell": n_sell,
            # "trd_n_tp": n_tp,
            # "trd_n_sl": n_sl,
            "trd_buy_sell_ratio": buy_sell_ratio,
            # "trd_stop_ratio": stop_ratio,
            "trd_net_dir_norm": net_dir_norm,
        })

    trd_feat = g_trd.apply(build_trade_features)
    trd_feat.index.name = "date"
    trd_feat = trd_feat.reset_index()
else:
    trd_feat = pd.DataFrame(columns=["date"])

# --- 2.4 合并所有日度特征 ---
feat_df = nav_df[["date", "ret_total"]].copy()
feat_df = feat_df.merge(cs_feat, on="date", how="left")
feat_df = feat_df.merge(pos_feat, on="date", how="left")
feat_df = feat_df.merge(trd_feat, on="date", how="left")

feat_df = feat_df.sort_values("date").reset_index(drop=True)
feat_df = feat_df.fillna(0.0)

# ===================== 3. 构造标签（基于下一天的收益） =====================
feat_df["ret_next"] = feat_df["ret_total"].shift(-1)
feat_df["y"] = (feat_df["ret_next"] > -0.03).astype(int)

# 最后一天没有 ret_next，不能用来训练和预测
feat_df = feat_df.iloc[:-1].copy()

# 选特征列
drop_cols = ["date", "ret_total", "ret_next", "y"]
feature_cols = [c for c in feat_df.columns if c not in drop_cols]

X_full = feat_df[feature_cols].values.astype(float)
y_full = feat_df["y"].values.astype(int)
dates_full = feat_df["date"].values  # 与 X_full / y_full 一一对应，都是“t 日特征，对应 label t+1”

n_samples = len(feat_df)

print(f"总样本数(可用于构造X,y): {n_samples}")

# ===================== 4. 滚动训练 + 生成全时间段 signal_t =====================

# 我们要为 nav_df 的每一天生成一个 signal_t（t 日收盘的决策）
signal_series = pd.DataFrame({
    "date": nav_df["date"],
    "signal": 1,   # 先全部设为 1（全持仓），后面用模型期覆盖可预测区间
})

# 记录每次重训的简单验证效果，便于查看
rolling_stats = []

# 从最早可训练的下标开始：
# 注意：
#  - X_full 的 index i 对应日期 dates_full[i]，标签是 ret_next(t+1)
#  - 若用窗口 [i_train_start, i_train_end) 训练，那么能预测的“当天特征”下标从 i_pred_start 开始
# 一个简单做法：从 i = train_window 开始，每隔 refit_step 重训一次模型，
# 用当前模型对 [i, i+refit_step) 这段日期做预测。
start_idx = train_window
if start_idx >= n_samples:
    raise ValueError(f"样本太少，train_window={train_window} 无法满足滚动训练。总样本={n_samples}")

# 存放所有 (date, signal) 的列表（只覆盖模型可预测的那些天）
pred_records = []

i = start_idx
model_id = 0
while i < n_samples:
    model_id += 1
    # 训练集区间：[i - train_window, i)
    train_start = i - train_window
    train_end = i

    X_train = X_full[train_start:train_end]
    y_train = y_full[train_start:train_end]
    dates_train = dates_full[train_start:train_end]

    # 将当前这次模型负责预测的区间：[i, i + refit_step) ∩ [0, n_samples)
    pred_start = i
    pred_end = min(i + refit_step, n_samples)

    X_pred = X_full[pred_start:pred_end]
    dates_pred = dates_full[pred_start:pred_end]
    y_true_pred = y_full[pred_start:pred_end]

    # 标准化 + 模型训练
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_pred_std = scaler.transform(X_pred)

    clf = LogisticRegression(max_iter=1000, class_weight=None)
    clf.fit(X_train_std, y_train)

    # 预测
    p_pred = clf.predict_proba(X_pred_std)[:, 1]
    pred_label = (p_pred > 0.5).astype(int)
    acc = (pred_label == y_true_pred).mean() if len(y_true_pred) > 0 else np.nan

    rolling_stats.append({
        "model_id": model_id,
        "train_start_date": dates_train[0],
        "train_end_date": dates_train[-1],
        "pred_start_date": dates_pred[0] if len(dates_pred) > 0 else None,
        "pred_end_date": dates_pred[-1] if len(dates_pred) > 0 else None,
        "train_size": len(X_train),
        "pred_size": len(X_pred),
        "val_acc": acc,
    })

    print(f"[Model {model_id}] "
          f"Train {dates_train[0]} ~ {dates_train[-1]} "
          f"Predict {dates_pred[0]} ~ {dates_pred[-1]} "
          f"Acc={acc:.4f}")

    # 记录预测结果到列表
    for d, prob in zip(dates_pred, p_pred):
        sig = 1 if prob > 0.5 else 0
        pred_records.append((pd.to_datetime(d), sig))

    # 滚动
    i += refit_step

# 把滚动预测结果整理成 DataFrame，并按 date 聚合（同一天多次覆盖的话，取最近那次）
sig_val_df = pd.DataFrame(pred_records, columns=["date", "signal"])
if not sig_val_df.empty:
    sig_val_df = sig_val_df.sort_values("date")
    sig_val_df = sig_val_df.groupby("date", as_index=False)["signal"].last()
else:
    sig_val_df = pd.DataFrame(columns=["date", "signal"])

# 把模型可预测区间的 signal 覆盖到全时间序列上
signal_series = signal_series.merge(
    sig_val_df,
    on="date",
    how="left",
    suffixes=("_default", "_model")
)
signal_series["signal"] = signal_series["signal_model"].fillna(signal_series["signal_default"]).astype(int)
signal_series = signal_series[["date", "signal"]].sort_values("date").reset_index(drop=True)

# 合并回 nav_df
nav_merged = nav_df.merge(signal_series, on="date", how="left")
nav_merged["signal"] = nav_merged["signal"].fillna(1).astype(int)

# ===================== 5. 成本逻辑（口径 B） =====================
# 定义:
# signal_t      : t 日收盘的决策
# signal_prev_t : signal_{t-1}（第一天等于第一天的 signal）
# hold_flag_t   : 今天是否持仓 = signal_prev_t
# ret_total_prev: 原始策略当日收益（含底层成本）
# ret_total     : 叠加择时 + 额外进出场成本后的当日收益
# nav_prev      : 原始策略净值
# nav           : 叠加择时后的净值

nav_merged["ret_total_prev"] = nav_merged["ret_total"]
nav_merged["nav_prev"] = nav_merged["nav"]

# 构造 signal_prev
nav_merged["signal_prev"] = nav_merged["signal"].shift(1)
nav_merged.loc[0, "signal_prev"] = nav_merged.loc[0, "signal"]
nav_merged["signal_prev"] = nav_merged["signal_prev"].astype(int)

# 今天是否持仓：看昨天收盘的决定
nav_merged["hold_flag"] = nav_merged["signal_prev"]

# 基础收益：若持仓则取原始收益，否则为 0
nav_merged["ret_total"] = np.where(
    nav_merged["hold_flag"] == 1,
    nav_merged["ret_total_prev"],
    0.0
)

# 成本脉冲数组
n_nav = len(nav_merged)
cost_pulse = np.zeros(n_nav, dtype=float)

signal = nav_merged["signal"].values
signal_prev_arr = nav_merged["signal_prev"].values

for i in range(n_nav):
    if i == 0:
        continue

    # 昨天 signal_{t-1} = signal_prev_arr[i]，今天 signal_t = signal[i]

    # 1) 平仓成本：昨天 1，今天 0 → 今天是平仓日
    if signal_prev_arr[i] == 1 and signal[i] == 0:
        cost_pulse[i] -= sell_cost_total

    # 2) 建仓成本：昨天 0，今天 1 → 明天是建仓日
    if signal_prev_arr[i] == 0 and signal[i] == 1:
        if i + 1 < n_nav:
            cost_pulse[i + 1] -= buy_cost_total
        # 超出末尾就忽略（没有明天的收益）

nav_merged["ret_total"] += cost_pulse
nav_merged["nav"] = (1.0 + nav_merged["ret_total"]).cumprod()
# 在成本部分之后临时加：
trans_in  = np.sum((nav_merged["signal_prev"] == 0) & (nav_merged["signal"] == 1))
trans_out = np.sum((nav_merged["signal_prev"] == 1) & (nav_merged["signal"] == 0))
print("开仓次数:", trans_in, "平仓次数:", trans_out)
# ===================== 6. 列名和顺序调整 =====================
final_cols = [
    "date",
    "ret_long",
    "ret_short",
    "ret_total_prev",
    "nav_prev",
    "n_long",
    "signal",
    "signal_prev",
    "hold_flag",
    "ret_total",
    "nav",
]

# 如果有列缺失可在这里按需删掉或替换
nav_out = nav_merged[final_cols].copy()

print("\n[Check] 最后三行：")
print(nav_out.tail(3))

# ===================== 7. 性能指标计算（可选） =====================
ret_base = nav_out["ret_total_prev"].values
ret_new = nav_out["ret_total"].values

def annual_return(ret_series, freq=252):
    mean_daily = np.nanmean(ret_series)
    return (1 + mean_daily) ** freq - 1

def annual_vol(ret_series, freq=252):
    return np.nanstd(ret_series, ddof=1) * np.sqrt(freq)

def sharpe(ret_series, freq=252, rf=0.0):
    ar = annual_return(ret_series, freq)
    av = annual_vol(ret_series, freq)
    if av == 0:
        return np.nan
    return (ar - rf) / av

print("\n=== 原始策略表现（含底层成本） ===")
print(f"年化收益: {annual_return(ret_base):.4f}")
print(f"年化波动: {annual_vol(ret_base):.4f}")
print(f"Sharpe : {sharpe(ret_base):.4f}")

print("\n=== 叠加滚动择时 + 进出场成本后的策略表现 ===")
print(f"年化收益: {annual_return(ret_new):.4f}")
print(f"年化波动: {annual_vol(ret_new):.4f}")
print(f"Sharpe : {sharpe(ret_new):.4f}")

hold_ratio = (nav_out["hold_flag"] == 1).mean()
print(f"\n平均持仓天数比例: {hold_ratio:.4f}")

# 打印每次滚动训练的简单验证结果
print("\n=== 每次滚动模型的 out-of-sample 准确率 ===")
rolling_stats_df = pd.DataFrame(rolling_stats)
print(rolling_stats_df)

# ===================== 8. 保存结果 =====================
out_path = base_dir / f"nav_with_timing_rolling_{run_name}.csv"
nav_out.to_csv(out_path, index=False, float_format="%.8f")
print(f"\n[OK] 已保存含滚动择时与成本修正的新净值到: {out_path}")