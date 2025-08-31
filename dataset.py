# coding: utf-8
"""
PyTorch Dataset：时序特征 + 行业 id + 板块 id
"""
import h5py, pickle, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from pathlib import Path

from utils import mad_clip, load_universe
from config import CFG


class StockDataset(Dataset):
    def __init__(self,
                 features_path: Path,
                 label_path: Path,
                 scaler_path: Path,
                 universe_path: Path):

        # ---------- 标签 ----------
        self.labels = pd.read_parquet(label_path)

        # ---------- 标准化器 ----------
        with open(scaler_path, "rb") as f:
            scalers = pickle.load(f)
            self.daily_scaler = scalers["daily"]
            self.min30_scaler = scalers["min30"]

        # ---------- 行业 / 板块映射 ----------
        self._build_mappings()

        # ---------- 股票池 ----------
        self.universe = load_universe(universe_path)

        # ---------- 样本索引 ----------
        self.features_path = features_path
        self._build_index()

    # ------------------------------------------------------------------
    def _build_mappings(self):
        # 行业
        if not CFG.industry_map_file.exists():
            self.ind2id, self.stk2ind = {"未知": 0}, {}
        else:
            ind_df = pd.read_csv(CFG.industry_map_file).dropna(subset=['industry'])
            ind_vals = sorted(ind_df["industry"].astype(str).unique())
            self.ind2id = {v: i for i, v in enumerate(ind_vals)}
            self.ind2id.setdefault("未知", len(self.ind2id))
            self.stk2ind = ind_df.set_index("stock")["industry"].astype(str).to_dict()
        self.num_industries = len(self.ind2id)

        # 板块
        if not CFG.sector_map_file.exists():
            self.sec2id, self.stk2sec = {"未知": 0}, {}
        else:
            sec_df = pd.read_csv(CFG.sector_map_file).dropna(subset=['sector'])
            sec_vals = sorted(sec_df["sector"].astype(str).unique())
            self.sec2id = {v: i for i, v in enumerate(sec_vals)}
            self.sec2id.setdefault("未知", len(self.sec2id))
            self.stk2sec = sec_df.set_index("stock")["sector"].astype(str).to_dict()
        self.num_sectors = len(self.sec2id)

    # ------------------------------------------------------------------
    def _build_index(self):
        """遍历 h5 构建样本索引"""
        self.idxs = []
        with h5py.File(self.features_path, "r") as h5f:
            for g_key in h5f.keys():
                date = pd.Timestamp(h5f[g_key].attrs["date"])
                stocks = h5f[g_key]["stocks"][:].astype(str).tolist()
                for idx, stk in enumerate(stocks):
                    if (date, stk) in self.labels.index:
                        self.idxs.append(dict(g_key=g_key,
                                              date=date,
                                              idx=idx,
                                              stock=stk))

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.idxs)

    # ------------------------------------------------------------------
    def __getitem__(self, i: int):
        meta = self.idxs[i]

        # -------- 读取特征 --------
        with h5py.File(self.features_path, "r") as h5f:
            grp = h5f[meta["g_key"]]
            daily_arr = grp["daily"][meta["idx"]]   # [Td,6]
            min30_arr = grp["min30"][meta["idx"]]   # [Tm,6]

        # -------- 预处理 --------
        daily_arr  = self._process_modality(daily_arr,  self.daily_scaler)
        min30_arr  = self._process_modality(min30_arr,  self.min30_scaler)

        # -------- 行业/板块 id --------
        ind = self.stk2ind.get(meta["stock"], "未知")
        sec = self.stk2sec.get(meta["stock"], "未知")
        ind_id = self.ind2id.get(ind, self.ind2id["未知"])
        sec_id = self.sec2id.get(sec, self.sec2id["未知"])

        # -------- 标签 --------
        label = self.labels.loc[(meta["date"], meta["stock"]), "next_week_return"]

        date_int = int(meta["date"].strftime('%Y%m%d'))

        return dict(
            daily=torch.tensor(daily_arr, dtype=torch.float32),
            min30=torch.tensor(min30_arr, dtype=torch.float32),
            ind_id=torch.tensor(ind_id, dtype=torch.long),
            sec_id=torch.tensor(sec_id, dtype=torch.long),
            label=torch.tensor(label, dtype=torch.float32),
            date=torch.tensor(date_int, dtype=torch.int32)
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _process_modality(arr: np.ndarray, scaler):
        """
        MAD 去极值 + Scaler + NaN/Inf → 0
        """
        if np.isnan(arr).all():
            return np.zeros_like(arr, dtype=np.float32)

        arr = mad_clip(arr)
        arr = scaler.transform(arr[None])[0]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr.astype(np.float32)


# ---------------- collate_fn ----------------
def collate_fn(batch):
    return dict(
        daily=torch.stack([b["daily"] for b in batch]),
        min30=torch.stack([b["min30"] for b in batch]),
        ind_id=torch.stack([b["ind_id"] for b in batch]),
        sec_id=torch.stack([b["sec_id"] for b in batch]),
        label=torch.stack([b["label"] for b in batch]),
        date=torch.stack([b["date"] for b in batch])
    )