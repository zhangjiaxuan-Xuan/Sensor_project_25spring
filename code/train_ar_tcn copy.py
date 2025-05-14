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

import data_extor  # 读取模块

# ——————————————————————————————
# 参数定义
WAIST   = ["左腰","中间","右腰","后腰"]
LEG     = ["左腿","右腿"]
FS      = 100.0        # Hz
WINDOW  = 50           # 过去窗口长度
HORIZON = 5            # 未来预测步数
FEAT_IN = 9            # 单IMU特征数
D_IN    = FEAT_IN * len(WAIST)
D_OUT   = FEAT_IN * len(LEG)

# ——————————————————————————————
# 滑窗构造

def build_window_seq(folder_dict: Dict[str, pd.DataFrame]):
    common_t = None
    for df in folder_dict.values():
        tt = df['time'].values
        common_t = tt if common_t is None else np.intersect1d(common_t, tt)
    common_t.sort()
    arr = {k: folder_dict[k].set_index('time').loc[common_t].to_numpy(dtype=np.float32)
           for k in folder_dict}
    Xs, Ys = [], []
    for i in range(WINDOW, len(common_t) - HORIZON):
        past = [arr[p][i-WINDOW:i,:FEAT_IN] for p in WAIST]
        Xs.append(np.concatenate(past,axis=1))
        seq = []
        for h in range(1, HORIZON+1):
            fut = [arr[p][i+h,:FEAT_IN] for p in LEG]
            seq.append(np.concatenate(fut))
        Ys.append(np.stack(seq,axis=0))
    return np.stack(Xs), np.stack(Ys)

# ——————————————————————————————
# DataLoader 构造

def get_dataloaders(root: str, batch: int):
    root = pathlib.Path(root)
    X_all, y_all, folders = [], [], []
    for fld in sorted(root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
        if fld.is_dir() and fld.name.isdigit():
            dd = data_extor.load_imu_folder(fld.name)
            if len(dd) >= len(WAIST)+len(LEG):
                X, y = build_window_seq(dd)
                X_all.append(X); y_all.append(y); folders.append(int(fld.name))
    X = np.concatenate(X_all,axis=0); y = np.concatenate(y_all,axis=0)
    Xt = torch.from_numpy(X); yt = torch.from_numpy(y)
    ds = TensorDataset(Xt,yt)
    bounds = np.cumsum([0]+[len(x) for x in X_all])
    idx_map = {f:list(range(s,e)) for f,(s,e) in zip(folders,zip(bounds,bounds[1:]))}
    train_idx = sum([idx_map.get(i,[]) for i in (1,2,3,4,5)],[])
    val_idx   = idx_map.get(6,[])
    test_idx  = idx_map.get(7,[])
    def mk(idx): return DataLoader(torch.utils.data.Subset(ds,idx), batch_size=batch, shuffle=False)
    return mk(train_idx), mk(val_idx), mk(test_idx)

# ——————————————————————————————
# 自回归 TCN + 差分 预测 网络

class AutoRegressiveTCN(nn.Module):
    def __init__(self, D_in=D_IN, D_out=D_OUT, H=HORIZON, hidden=128, levels=4, k=3):
        super().__init__()
        self.H = H
        layers=[]; ch=D_in
        for i in range(levels):
            dil=2**i; pad=(k-1)*dil
            layers += [
                nn.Conv1d(ch,hidden,k,padding=pad,dilation=dil),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden,hidden,1),
                nn.ReLU(inplace=True)
            ]
            ch=hidden
        self.tcn = nn.Sequential(*layers)
        self.pool= nn.AdaptiveAvgPool1d(1)
        self.delta = nn.Linear(hidden,D_out)

    def forward(self, x, y0):
        # x: B,T,D_in → B,D_in,T
        h = x.permute(0,2,1)
        h = self.tcn(h)
        h = self.pool(h).squeeze(-1)
        y_prev = y0
        seq=[]
        for _ in range(self.H):
            d = self.delta(h)
            y_next = y_prev + d
            seq.append(y_next.unsqueeze(1)); y_prev=y_next
        return torch.cat(seq,dim=1)

# ——————————————————————————————
# 训练 + 调度 + 提前结束 + 最佳保存

def train_and_plot(root, batch, epochs, lr, outdir, patience: int = 50, factor: int = 0.5,step_size: int = 50, gamma: int = 0.5):
    outdir = pathlib.Path(outdir); outdir.mkdir(exist_ok=True,parents=True)
    tr, va, te = get_dataloaders(root,batch)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoRegressiveTCN().to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    # 使用阶段性衰减学习率 (StepLR)，每 50 个 epoch LR 降至原来的 0.1 倍
    scheduler = optim.lr_scheduler.StepLR(opt, step_size, gamma)  
    loss_fn = nn.MSELoss()

    best_val=math.inf; best_st=None; wait=0
    train_hist=[]; val_hist=[]
    for ep in range(1,epochs+1):
        # train
        model.train(); tr_loss=0; cnt=0
        for X,y in tr:
            X,y = X.to(device), y.to(device)
            y0 = y[:,0,:]
            opt.zero_grad(); pred=model(X,y0)
            loss=loss_fn(pred,y); loss.backward(); opt.step()
            tr_loss+=loss.item()*X.size(0); cnt+=X.size(0)
        tr_loss/=max(cnt,1)
        # val
        model.eval(); val_loss=0; cnt=0
        with torch.no_grad():
            for X,y in va:
                X,y=X.to(device),y.to(device); y0=y[:,0,:]
                l=loss_fn(model(X,y0),y).item()
                val_loss+=l*X.size(0); cnt+=X.size(0)
        val_loss/=max(cnt,1)
        train_hist.append(tr_loss); val_hist.append(val_loss)
        print(f"Epoch {ep}: train={tr_loss:.4f}, val={val_loss:.4f}")
        # 按 epoch 固定步长更新学习率
        scheduler.step()
        # early stopping
        if val_loss<best_val:
            best_val=val_loss; best_st=model.state_dict(); wait=0
        else:
            wait+=1
            if wait>=patience:
                print(f"Early stopping at epoch {ep}")
                break
    # save best
    torch.save(best_st, outdir/'model_best.pt')
    # plot
    plt.figure(); plt.plot(train_hist,label='train'); plt.plot(val_hist,label='val');
    plt.legend(); plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.savefig(outdir/'loss_curve.png'); plt.close()
    # sample pred
    model.load_state_dict(best_st); model.eval()
    X,y = next(iter(te)); X=X.to(device); y0=y[:,0,:].to(device)
    pred = model(X,y0).cpu().detach().numpy()[0]; gt=y.numpy()[0]
    plt.figure(figsize=(8,3)); plt.plot(gt.flatten(),label='GT'); plt.plot(pred.flatten(),'--',label='Pred');
    plt.legend(); plt.title('AutoReg TCN prediction'); plt.savefig(outdir/'sample_pred.png'); plt.close()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument("--root", default=pathlib.Path(__file__).resolve().parent.parent, help="数据根目录(1-7 子文件夹)")
    p.add_argument('--batch',type=int,default=64)
    p.add_argument('--epochs',type=int,default=3000)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--out',default='runs_diff')
    p.add_argument('--patience',type=int,default=150,help='early stopping patience')
    p.add_argument('--factor',type=float,default=0.5,help='lr scheduler factor')
    args=p.parse_args()
    train_and_plot(args.root,args.batch,args.epochs,args.lr,args.out,args.patience,args.factor)
