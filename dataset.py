# coding: utf-8
"""
多源特征数据集（仅日频因子 + 市场/风格上下文 + 行业 ID）
"""
import h5py, torch, numpy as np, pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from config import CFG
from utils import pkl_load


class MultiFeatureDataset(Dataset):
    """
    单个采样日期 = HDF5 中一个 group（形如 date_0 / date_1 ...）
    每条样本 = (日频因子序列, 行业 id, 上下文向量, 周度收益标签)
    """
    def __init__(
        self,
        daily_h5: Path,                 # features_daily.h5
        ctx_file: Path,                 # context_features.parquet
        label_df: pd.DataFrame,         # MultiIndex (date, stock) -> next_week_return
        scaler_daily,                   # utils.Scaler
        industry_map: dict,             # {stock: industry_id}
        date_indices                    # 需要读取的 group 名列表，例如 ["date_0", "date_1"]
    ):
        super().__init__()
        self.h5 = h5py.File(daily_h5, "r")
        self.scaler = scaler_daily
        self.lbl = label_df
        self.ind_map = industry_map

        # 读取上下文特征 DataFrame（index = 周五日期）
        self.ctx_df = pd.read_parquet(ctx_file)
        self.ctx_dim = self.ctx_df.shape[1]

        self.factor_cols = self.h5.attrs["factor_cols"].astype(str)

        # ------- 构造索引 ------- #
        # 每个元素: (group_key, row_id, date, stock)
        self.idx = []
        for gk in date_indices:
            if gk not in self.h5:
                continue
            g = self.h5[gk]
            date = pd.Timestamp(g.attrs["date"])
            stocks = g["stocks"][:].astype(str)

            for i, stk in enumerate(stocks):
                if (date, stk) in self.lbl.index:
                    self.idx.append((gk, i, date, stk))

        if len(self.idx) == 0:
            raise RuntimeError("当前日期窗口未匹配到任何标签样本")

    # -------------------- Dataset API -------------------- #
    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        gk, row, date, stk = self.idx[i]

        # 日频因子 [T, C] -> 标准化
        feat = self.h5[gk]["factor"][row]          # 期望是 numpy [T,C]
        # 防御性处理：去掉多余的维度（如 [T,1,C] 或 [T,C,1]）
        feat = np.asarray(feat)
        # --------- 统一去掉所有 size==1 的维度 ---------
        feat = np.squeeze(feat)                    # 得到 (T,C)

        feat = self.scaler.transform(feat)         # 标准化仍为 [T, C]
        feat = np.squeeze(feat)  
        if feat.ndim != 2:
            raise RuntimeError(f"factor shape 期望 [T,C]，当前 {feat.shape}")

        # 行业 id（未知置为 padding_id = n_ind）
        ind_id = self.ind_map.get(stk, self.pad_ind_id)

        # 上下文向量（同一日期全股票一致；无则 0）
        if date in self.ctx_df.index:
            ctx_vec = self.ctx_df.loc[date].values.astype(np.float32)
        else:
            ctx_vec = np.zeros(self.ctx_dim, dtype=np.float32)

        # 标签
        y = self.lbl.loc[(date, stk), "next_week_return"]

        return (
            torch.tensor(feat, dtype=torch.float32),          # [T,C]
            torch.tensor(ind_id, dtype=torch.long),           # []
            torch.tensor(ctx_vec, dtype=torch.float32),       # [C_ctx]
            torch.tensor(y, dtype=torch.float32)              # []
        )
    # --------- 行业类别数 / padding id --------- #
    @property
    def n_ind(self):
        if not hasattr(self, "_n_ind"):
            inds = set(self.ind_map.values())
            self._n_ind = max(inds) + 1
        return self._n_ind

    @property
    def pad_ind_id(self):
        return self.n_ind   # 未知行业放最后一类


# ---------------- Mini-batch 组装 ---------------- #
def collate(batch):
    """
    输出:
        daily_feat : [B,T,C]
        ind_id     : [B]
        ctx_feat   : [B,C_ctx]
        y          : [B]
    """
    daily_feat = torch.stack([b[0] for b in batch])
    ind_id     = torch.stack([b[1] for b in batch])
    ctx_feat   = torch.stack([b[2] for b in batch])
    y          = torch.stack([b[3] for b in batch])
    return daily_feat, ind_id, ctx_feat, y