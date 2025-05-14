#!/usr/bin/env python3
# train_ar_tcn.py

import argparse, pathlib, math
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

import data_extor  # 你现有的数据加载模块

# ——————————————————————————————
# Ⅰ. 参数定义
WAIST = ["左腰", "中间", "右腰", "后腰"]
LEG   = ["左腿", "右腿"]
FS     = 100.0        # Hz
WINDOW = 50           # 0.50 s 过去窗口
HORIZON= 5            # 0.05 s 未来序列长度
FEAT_IN= 9            # 每个 IMU 特征数
D_IN   = FEAT_IN * len(WAIST)
D_OUT  = FEAT_IN * len(LEG)

# ——————————————————————————————
# Ⅱ. 滑窗构造（多步预测版）
def build_window_seq(folder_dict: Dict[str, pd.DataFrame]):
    """
    X: (N, WINDOW, D_IN),  y: (N, HORIZON, D_OUT)
    """
    # 1) 找公共时间轴
    common_t = None
    for df in folder_dict.values():
        tt = df["time"].values
        common_t = tt if common_t is None else np.intersect1d(common_t, tt)
    common_t.sort()

    # 2) 构建 time → array lookup
    arr = {
        name: folder_dict[name].set_index("time").loc[common_t].to_numpy(dtype=np.float32)
        for name in folder_dict
    }

    Xs, Ys = [], []
    for i in range(WINDOW, len(common_t) - HORIZON):
        # 过去腰带数据拼接
        past = [arr[p][i-WINDOW:i, :FEAT_IN] for p in WAIST]
        Xs.append(np.concatenate(past, axis=1))  # (WINDOW, D_IN)
        # 未来腿部序列
        seq = []
        for h in range(1, HORIZON+1):
            fut = [arr[p][i+h, :FEAT_IN] for p in LEG]
            seq.append(np.concatenate(fut))      # (D_OUT,)
        Ys.append(np.stack(seq, axis=0))        # (HORIZON, D_OUT)

    return np.stack(Xs), np.stack(Ys)

# ——————————————————————————————
# Ⅲ. DataLoader 构造
def get_dataloaders(root: str, batch: int):
    root = pathlib.Path(root)
    X_all, y_all, folders = [], [], []
    for fld in sorted(root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
        if not fld.is_dir() or not fld.name.isdigit(): continue
        dd = data_extor.load_imu_folder(fld.name)
        if len(dd) < len(WAIST)+len(LEG): continue
        X, y = build_window_seq(dd)
        X_all.append(X); y_all.append(y); folders.append(int(fld.name))

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    Xt = torch.from_numpy(X)    # (N, WINDOW, D_IN)
    yt = torch.from_numpy(y)    # (N, HORIZON, D_OUT)
    ds = TensorDataset(Xt, yt)

    # 按文件夹分 idx
    bounds = np.cumsum([0] + [len(x) for x in X_all])
    idx_map = {f: list(range(s, e)) for f,(s,e) in zip(folders, zip(bounds, bounds[1:]))}

    train_idx = sum([idx_map.get(i,[]) for i in (1,2,3,4,5)], [])
    val_idx   = idx_map.get(6, [])
    test_idx  = idx_map.get(7, [])

    def mk(idx): 
        return DataLoader(torch.utils.data.Subset(ds, idx), batch_size=batch, shuffle=False)
    return mk(train_idx), mk(val_idx), mk(test_idx)

# ——————————————————————————————
# Ⅳ. 自回归 TCN + 差分 预测 网络
class AutoRegressiveTCN(nn.Module):
    def __init__(self, D_in=D_IN, D_out=D_OUT, H=HORIZON, hidden=64, levels=4, k=3):
        super().__init__()
        self.H = H
        # TCN 主干
        layers = []; ch = D_in
        for i in range(levels):
            dil = 2**i; pad = (k-1)*dil
            layers += [
                nn.Conv1d(ch, hidden, k, padding=pad, dilation=dil),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden, hidden, 1),
                nn.ReLU(inplace=True)
            ]
            ch = hidden
        self.tcn = nn.Sequential(*layers)
        self.pool= nn.AdaptiveAvgPool1d(1)
        self.delta = nn.Linear(hidden, D_out)

    def forward(self, x, y0):
        # x: (B,T,D_in) → (B,D_in,T)
        h = x.permute(0,2,1)
        h = self.tcn(h)
        h = self.pool(h).squeeze(-1)  # (B, hidden)
        y_prev = y0                     # (B, D_out)
        seq = []
        for _ in range(self.H):
            d = self.delta(h)           # 增量
            y_next = y_prev + d
            seq.append(y_next.unsqueeze(1))
            y_prev = y_next
        return torch.cat(seq, dim=1)    # (B,H,D_out)

# ——————————————————————————————
# Ⅴ. 训练 + 可视化
def train_and_plot(root, batch, epochs, lr, outdir):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tr, va, te = get_dataloaders(root, batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoRegressiveTCN().to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = math.inf; best_st=None
    train_hist=[]; val_hist=[]

    for ep in range(1, epochs+1):
        # train one epoch
        model.train(); lt=0; n=0
        for X,y in tr:
            X,y = X.to(device), y.to(device)
            opt.zero_grad()
            y0 = y[:,0,:]
            pred = model(X, y0)      # (B,H,D_out)
            loss = loss_fn(pred, y)
            loss.backward(); opt.step()
            lt += loss.item()*X.size(0); n += X.size(0)
        lt /= max(n,1)

        # val
        model.eval(); lv=0; n=0
        with torch.no_grad():
            for X,y in va:
                X,y = X.to(device), y.to(device)
                y0 = y[:,0,:]
                lv  += loss_fn(model(X,y0), y).item()*X.size(0)
                n   += X.size(0)
        lv /= max(n,1)

        train_hist.append(lt); val_hist.append(lv)
        print(f"Epoch {ep}: train={lt:.4f}, val={lv:.4f}")
        if lv<best_val:
            best_val, best_st = lv, model.state_dict()

    # 保存
    od = pathlib.Path(outdir); od.mkdir(parents=True, exist_ok=True)
    torch.save(best_st, od/"model_best.pt")
    # 损失曲线
    plt.plot(train_hist, label="train")
    plt.plot(val_hist, label="val"); plt.legend()
    plt.xlabel("Epoch"); plt.ylabel("MSE")
    plt.savefig(od/"loss_curve.png"); plt.close()

    # 画示例预测 vs GT
# --- 预测示例 ---
    model.load_state_dict(best_st)
    model.eval()
    X, y = next(iter(te))
    X = X.to(device)
    #if i == 0: break
    y0 = y[:, 0, :].to(device)          # ← 确保 y0 在同一设备
    pred = model(X, y0).cpu().detach().numpy()[0]
    gt   = y.numpy()[0]

    plt.figure(figsize=(8,3))
    plt.plot(gt.flatten(), label="GT")
    plt.plot(pred.flatten(), "--", label="Pred")
    plt.legend(); plt.title("AutoReg TCN prediction")
    plt.savefig(outdir/f"sample_pred_{0}.png"); plt.close()

# ——————————————————————————————
# Ⅵ. CLI
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root", default=pathlib.Path(__file__).resolve().parent.parent, help="数据根目录(1-7 子文件夹)")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs",type=int,default=300)
    p.add_argument("--lr",   type=float,default=1e-3)
    p.add_argument("--out",  default="runs")
    args = p.parse_args()
    train_and_plot(args.root, args.batch, args.epochs, args.lr, args.out)
