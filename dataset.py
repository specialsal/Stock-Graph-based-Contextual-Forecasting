# coding: utf-8
"""
PyTorch Dataset：返回时序特征 + 行业 id + 板块 id
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
        # ---------------- 基本路径 ----------------
        self.features_path = features_path
        self.labels = pd.read_parquet(label_path)

        # ---------------- 标准化器 ----------------
        with open(scaler_path, "rb") as f:
            scalers = pickle.load(f)
            self.daily_scaler = scalers["daily"]
            self.min30_scaler = scalers["min30"]

        # ---------------- 股票池 ----------------
        self.universe = load_universe(universe_path)

        # ---------------- 建立股票→行业/板块映射 ----------------
        # 检查文件是否存在
        if not CFG.industry_map_file.exists():
            print(f"警告: 行业映射文件不存在: {CFG.industry_map_file}")
            self.ind2id = {"未知": 0}
            self.stk2ind = {}
            self.num_industries = 1
        else:
            ind_df = pd.read_csv(CFG.industry_map_file)
            print(f"行业映射文件形状: {ind_df.shape}")
            print(f"行业映射文件列名: {ind_df.columns.tolist()}")
            print(f"行业列数据类型: {ind_df['industry'].dtype}")
            print(f"行业列前5个值: {ind_df['industry'].head().tolist()}")
            print(f"行业列是否有NaN: {ind_df['industry'].isna().sum()}")

            # 安全处理行业数据
            ind_df = ind_df.dropna(subset=['industry'])  # 删除industry为NaN的行
            ind_values = ind_df["industry"].unique()
            # 确保所有值都是字符串类型
            ind_values = [str(x) for x in ind_values if pd.notna(x)]
            ind_list = sorted(ind_values)
            self.ind2id = {ind: i for i, ind in enumerate(ind_list)}
            # 添加默认值
            if "未知" not in self.ind2id:
                self.ind2id["未知"] = len(self.ind2id)

            self.stk2ind = ind_df.set_index("stock")["industry"].to_dict()
            self.num_industries = len(self.ind2id)

        if not CFG.sector_map_file.exists():
            print(f"警告: 板块映射文件不存在: {CFG.sector_map_file}")
            self.sec2id = {"未知": 0}
            self.stk2sec = {}
            self.num_sectors = 1
        else:
            sec_df = pd.read_csv(CFG.sector_map_file)
            print(f"板块映射文件形状: {sec_df.shape}")
            print(f"板块映射文件列名: {sec_df.columns.tolist()}")
            print(f"板块列数据类型: {sec_df['sector'].dtype}")
            print(f"板块列前5个值: {sec_df['sector'].head().tolist()}")
            print(f"板块列是否有NaN: {sec_df['sector'].isna().sum()}")

            # 安全处理板块数据
            sec_df = sec_df.dropna(subset=['sector'])  # 删除sector为NaN的行
            sec_values = sec_df["sector"].unique()
            # 确保所有值都是字符串类型
            sec_values = [str(x) for x in sec_values if pd.notna(x)]
            sec_list = sorted(sec_values)
            self.sec2id = {sec: i for i, sec in enumerate(sec_list)}
            # 添加默认值
            if "未知" not in self.sec2id:
                self.sec2id["未知"] = len(self.sec2id)

            self.stk2sec = sec_df.set_index("stock")["sector"].to_dict()
            self.num_sectors = len(self.sec2id)

        print(f"加载完成 - 行业数量: {self.num_industries}, 板块数量: {self.num_sectors}")
        print(f"行业映射示例: {dict(list(self.ind2id.items())[:5])}")
        print(f"板块映射示例: {dict(list(self.sec2id.items())[:5])}")

        # ---------------- 构建样本索引 ----------------
        self._build_index()

    # -------------------------------------------------
    def _build_index(self):
        """遍历 h5 文件，生成可用样本索引列表"""
        self.idxs = []
        with h5py.File(self.features_path, "r") as h5f:
            for d_key in h5f.keys():  # 每个日期 group
                date = pd.Timestamp(h5f[d_key].attrs["date"])
                stocks = h5f[d_key]["stocks"][:].astype(str).tolist()

                for i, stk in enumerate(stocks):
                    if (date, stk) in self.labels.index:
                        self.idxs.append(dict(g_key=d_key, date=date, idx=i, stock=stk))

    # -------------------------------------------------
    def __len__(self):
        return len(self.idxs)

    # -------------------------------------------------
    def __getitem__(self, i: int):
        meta = self.idxs[i]

        # ---------- 读取时序特征 ----------
        with h5py.File(self.features_path, "r") as h5f:
            grp = h5f[meta["g_key"]]
            daily_arr = grp["daily"][meta["idx"]]  # [Td,6]
            min30_arr = grp["min30"][meta["idx"]]  # [Tm,6]

        # ---------- 标准化 + 去极值 ----------
        daily_arr = self._process_modality(daily_arr, self.daily_scaler)
        min30_arr = self._process_modality(min30_arr, self.min30_scaler)

        # ---------- 行业 / 板块 id ----------
        ind_raw = self.stk2ind.get(meta["stock"], None)
        sec_raw = self.stk2sec.get(meta["stock"], None)

        # 处理行业ID
        if pd.isna(ind_raw) or ind_raw is None:
            ind_id = self.ind2id.get("未知", 0)
        else:
            ind_str = str(ind_raw)  # 确保是字符串
            ind_id = self.ind2id.get(ind_str, self.ind2id.get("未知", 0))

        # 处理板块ID
        if pd.isna(sec_raw) or sec_raw is None:
            sec_id = self.sec2id.get("未知", 0)
        else:
            sec_str = str(sec_raw)  # 确保是字符串
            sec_id = self.sec2id.get(sec_str, self.sec2id.get("未知", 0))

        # ---------- 标签 ----------
        label = self.labels.loc[(meta["date"], meta["stock"]), "next_week_return"]

        return dict(
            daily=torch.tensor(daily_arr, dtype=torch.float32),
            min30=torch.tensor(min30_arr, dtype=torch.float32),
            ind_id=torch.tensor(ind_id, dtype=torch.long),
            sec_id=torch.tensor(sec_id, dtype=torch.long),
            label=torch.tensor(label, dtype=torch.float32)
        )

    # -------------------------------------------------
    @staticmethod
    def _process_modality(arr: np.ndarray, scaler):
        """MAD 去极值 + Z-Score 标准化"""
        if np.isnan(arr).all():
            return np.zeros_like(arr, dtype=np.float32)
        arr = mad_clip(arr)
        arr = scaler.transform(arr[None])[0]
        return arr.astype(np.float32)


# ---------------- collate_fn ----------------
def collate_fn(batch):
    return dict(
        daily=torch.stack([b["daily"] for b in batch]),  # [B,Td,6]
        min30=torch.stack([b["min30"] for b in batch]),  # [B,Tm,6]
        ind_id=torch.stack([b["ind_id"] for b in batch]),  # [B]
        sec_id=torch.stack([b["sec_id"] for b in batch]),  # [B]
        label=torch.stack([b["label"] for b in batch])  # [B]
    )