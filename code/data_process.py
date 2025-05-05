"""train_gait.py
============================================
可直接 `python train_gait.py --root DATA_ROOT` 运行的完整脚本：
从已有 `data_extor.load_imu_folder()` 加载数字文件夹 1‑7 →
→ 50‑帧腰带窗口预测 0.3 s 后 18‑维腿向量，
→ 双层 LSTM 训练，输出 loss 曲线与示例预测图。

命令行参数
-------------
--root   数据根目录（里面是数字文件夹 1‑7）
--batch  批大小（default 64）
--epochs 训练轮数（default 30）
--lr     学习率（default 1e-3）
--out    结果输出目录（default runs）
"""
from __future__ import annotations
import argparse, pathlib, math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

# =====  A.  外部数据读取  ========================================
import data_extor  # 直接调用用户已有的读取函数

WAIST = ["左腰", "中间", "右腰", "后腰"]
LEG   = ["左腿", "右腿"]
FS = 100.0
WINDOW   = 50   # 0.50 s 过去窗口
HORIZON  = 20   # 0.50 s 未来
WAIST_DIM = 9 * len(WAIST)  # 4*9=36
LEG_DIM   = 9 * len(LEG)    # 2*9=18
FEAT_IN = 9  # 每个 IMU 的特征数
# =====  B.  滑窗生成  ============================================

def build_window(folder_dict: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
    """将单文件夹所有 IMU DataFrame → (X,y)
    X: (N, WINDOW, 36)
    y: (N, 18)  (future horizon 的腿特征)
    """
    # 1. 找公共时间戳
    common_t = None
    for df in folder_dict.values():
        t = df["time"].values
        common_t = t if common_t is None else np.intersect1d(common_t, t)
    common_t.sort()

    # 2. 构建 lookup
    arr = {pl: folder_dict[pl].set_index("time").loc[common_t].to_numpy(dtype=np.float32)
           for pl in folder_dict}

    X_list, y_list = [], []
    start = WINDOW
    end   = len(common_t) - HORIZON
    for idx in range(start, end):
        # past waist features
        past = [arr[p][idx-WINDOW:idx, :9] for p in WAIST]
        X_list.append(np.concatenate(past, axis=1))               # (WINDOW,36)
        # future leg single frame
        fut = [arr[p][idx+HORIZON, :9] for p in LEG]
        y_list.append(np.concatenate(fut))                        # (18,)
    return np.stack(X_list), np.stack(y_list)

# =====  C.  DataLoader 构建  =====================================

def get_dataloaders(root:str|pathlib.Path, batch:int=64) -> Tuple[DataLoader,DataLoader,DataLoader]:
    root = pathlib.Path(root)
    X_all, y_all, folder_idx = [], [], []
    for fld in sorted(root.iterdir(), key=lambda p:int(p.name) if p.name.isdigit() else 0):
        if not fld.is_dir() or not fld.name.isdigit():
            continue
        dic = data_extor.load_imu_folder(fld.name)  # 使用用户函数
        if len(dic) < 6:  # 缺某些 IMU
            continue
        X, y = build_window(dic)
        X_all.append(X); y_all.append(y); folder_idx.append(int(fld.name))
    # 拼接
    X = np.concatenate(X_all); y=np.concatenate(y_all)
    X_t = torch.from_numpy(X)   # (N,50,36)
    y_t = torch.from_numpy(y)   # (N,18)
    dataset = TensorDataset(X_t, y_t)

    # index mapping per folder
    bounds = np.cumsum([0]+[len(x) for x in X_all])
    folder_ids = {}
    for i,(s,e) in enumerate(zip(bounds[:-1], bounds[1:])):
        folder_ids[folder_idx[i]] = list(range(s,e))

    train_idx = sum([folder_ids.get(n,[]) for n in (1,2,3,4,5)], [])
    val_idx   = folder_ids.get(6, [])
    test_idx  = folder_ids.get(7, [])

    def make_loader(idx):
        return DataLoader(torch.utils.data.Subset(dataset, idx), batch_size=batch, shuffle=bool(idx))

    return make_loader(train_idx), make_loader(val_idx), make_loader(test_idx)

# =====  D.  网络  =================================================

class GaitLSTM(nn.Module):
    def __init__(self, in_dim=WAIST_DIM, hidden=128, out_dim=LEG_DIM, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True)
        self.fc   = nn.Sequential(nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
    def forward(self, x):
        _,(h,_)=self.lstm(x)
        return self.fc(h[-1])

class MLP(nn.Module):
    def __init__(self, window: int = WINDOW, feat_per_imu: int = FEAT_IN, imu_count: int = 4, hidden1: int = 512, hidden2: int = 256, out_dim: int = FEAT_IN*2):
        super().__init__()
        in_dim = window * feat_per_imu * imu_count
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1), nn.ReLU(True),
            nn.Linear(hidden1, hidden2),    nn.ReLU(True),
            nn.Linear(hidden2, out_dim)
        )
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.net(x)

# =====  E.  训练与可视化  ==========================================

def train_and_plot(root:str, batch:int=64, epochs:int=30, lr:float=1e-3, outdir="runs"):
    train_loader, val_loader, test_loader = get_dataloaders(root, batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = GaitLSTM().to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_state=None; best_val=math.inf
    train_hist, val_hist = [], []

    for ep in range(1, epochs+1):
        # --- train ---
        model.train(); tr_loss=0; n=0
        for X,y in train_loader:
            X,y = X.to(device), y.to(device)
            opt.zero_grad(); pred=model(X); loss=loss_fn(pred,y)
            loss.backward(); opt.step()
            tr_loss += loss.item()*X.size(0); n += X.size(0)
        tr_loss /= max(n,1)

        # --- val ---
        model.eval(); val_loss=0; n=0
        with torch.no_grad():
            for X,y in val_loader:
                X,y = X.to(device), y.to(device)
                loss = loss_fn(model(X),y)
                val_loss+=loss.item()*X.size(0); n+=X.size(0)
        val_loss /= max(n,1)

        train_hist.append(tr_loss); val_hist.append(val_loss)
        print(f"Epoch {ep:02d}: train {tr_loss:.4f}  val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss; best_state = model.state_dict()

    # 保存结果
    out = pathlib.Path(outdir); out.mkdir(exist_ok=True, parents=True)
    torch.save(best_state, out/"model_best.pt")

    plt.figure(); plt.plot(train_hist,label="train"); plt.plot(val_hist,label="val"); plt.legend();
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("Loss curve"); plt.savefig(out/"loss_curve.png"); plt.close()

    # --- 预测示例 ---
    model.load_state_dict(best_state); model.eval()
    X,y = next(iter(test_loader))
    pred = model(X.to(device)).cpu().detach().numpy()[0]
    gt   = y.numpy()[0]
    plt.figure(figsize=(8,3)); plt.plot(gt,label="GT"); plt.plot(pred,"--",label="Pred"); plt.legend();
    plt.title("Leg vector prediction (first test sample)");
    plt.savefig(out/"sample_pred.png"); plt.close()

# =====  F.  CLI  ==================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=pathlib.Path(__file__).resolve().parent.parent, help="数据根目录，包含数字文件夹1‑7")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    train_and_plot(args.root, args.batch, args.epochs, args.lr, args.out)
