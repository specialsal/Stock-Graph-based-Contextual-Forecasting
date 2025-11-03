# coding: utf-8
"""
多源特征数据集（仅日频因子 + 市场/风格上下文 + 行业/风格ID）
本版：返回 chain_id, ind1_id, ind2_id 三路 id，适配新模型
"""
import h5py, torch, numpy as np, pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from config import CFG

class MultiFeatureDataset(Dataset):
    """
    单个采样日期 = HDF5 中一个 group（形如 date_0 / date_1 ...）
    每条样本 = (日频因子序列, chain_id, ind1_id, ind2_id, 上下文向量, 周度收益标签)
    """
    def __init__(
        self,
        daily_h5: Path,                 # features_daily.h5
        ctx_file: Path,                 # context_features.parquet
        label_df: pd.DataFrame,         # MultiIndex (date, stock) -> next_week_return
        scaler_daily,                   # utils.Scaler
        chain_map: dict,                # {stock: chain_id}
        ind1_map: dict,                 # {stock: ind1_id}
        ind2_map: dict,                 # {stock: ind2_id}
        date_indices                    # ["date_0", "date_1", ...]
    ):
        super().__init__()
        self.h5 = h5py.File(daily_h5, "r")
        self.scaler = scaler_daily
        self.lbl = label_df
        self.chain_map = chain_map
        self.ind1_map = ind1_map
        self.ind2_map = ind2_map

        self.ctx_df = pd.read_parquet(ctx_file)
        self.ctx_dim = self.ctx_df.shape[1]

        self.factor_cols = self.h5.attrs["factor_cols"].astype(str)

        # pad ids
        self._n_chain = (max(chain_map.values())+1) if len(chain_map)>0 else 0
        self._n_ind1  = (max(ind1_map.values())+1) if len(ind1_map)>0 else 0
        self._n_ind2  = (max(ind2_map.values())+1) if len(ind2_map)>0 else 0
        self.pad_chain = self._n_chain
        self.pad_ind1  = self._n_ind1
        self.pad_ind2  = self._n_ind2

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

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        gk, row, date, stk = self.idx[i]

        feat = np.asarray(self.h5[gk]["factor"][row])
        feat = np.squeeze(feat)
        feat = self.scaler.transform(feat)
        feat = np.squeeze(feat)
        if feat.ndim != 2:
            raise RuntimeError(f"factor shape 期望 [T,C]，当前 {feat.shape}")

        chain_id = self.chain_map.get(stk, self.pad_chain)
        ind1_id  = self.ind1_map.get(stk, self.pad_ind1)
        ind2_id  = self.ind2_map.get(stk, self.pad_ind2)

        if date in self.ctx_df.index:
            ctx_vec = self.ctx_df.loc[date].values.astype(np.float32)
        else:
            ctx_vec = np.zeros(self.ctx_dim, dtype=np.float32)

        y = self.lbl.loc[(date, stk), "next_week_return"]

        return (
            torch.tensor(feat, dtype=torch.float32),          # [T,C]
            torch.tensor(chain_id, dtype=torch.long),         # []
            torch.tensor(ind1_id, dtype=torch.long),          # []
            torch.tensor(ind2_id, dtype=torch.long),          # []
            torch.tensor(ctx_vec, dtype=torch.float32),       # [C_ctx]
            torch.tensor(y, dtype=torch.float32)              # []
        )

    @property
    def n_chain(self):
        return self._n_chain
    @property
    def n_ind1(self):
        return self._n_ind1
    @property
    def n_ind2(self):
        return self._n_ind2

def collate(batch):
    daily_feat = torch.stack([b[0] for b in batch])
    chain_id   = torch.stack([b[1] for b in batch])
    ind1_id    = torch.stack([b[2] for b in batch])
    ind2_id    = torch.stack([b[3] for b in batch])
    ctx_feat   = torch.stack([b[4] for b in batch])
    y          = torch.stack([b[5] for b in batch])
    return daily_feat, chain_id, ind1_id, ind2_id, ctx_feat, y