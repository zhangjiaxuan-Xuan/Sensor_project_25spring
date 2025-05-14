#!/usr/bin/env python3
"""show_stitched.py
====================
将测试集的多步窗口预测使用自回归 TCN 生成的 pred_windows
缝合成完整序列后，与真实值对齐，逐特征维度绘制 GT vs. Stitched‑Pred。
新增参数 --feature 可选绘制单个特征（0-8），默认为all即全部特征。

用法：
    python show_stitched.py \
        --root DATA_ROOT \
        --model runs/model_best.pt \
        --out viz \
        [--folder 7] [--feature all]
"""
import argparse, pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from train_ar_tcn import AutoRegressiveTCN, get_dataloaders, WINDOW, HORIZON, D_OUT

FS = 100

def stitch_windows(pred_windows: np.ndarray, total_length: int,
                   window_size: int = WINDOW, horizon: int = HORIZON) -> np.ndarray:
    """
    把 shape=(N, HORIZON, D_OUT) 的预测片段按时间缝合成完整序列
    返回 shape=(total_length, D_OUT) 的 stitched predictions
    """
    N, H, D = pred_windows.shape
    acc    = np.zeros((total_length, D), dtype=np.float32)
    counts = np.zeros((total_length, D), dtype=np.int32)
    # 第 i 个窗口的 h 步预测对应 time index = i+ h + 1
    for i in range(N):
        for h in range(H):
            t = i + h + 1
            if t < total_length:
                acc[t] += pred_windows[i, h]
                counts[t] += 1
    counts[counts == 0] = 1
    return acc / counts


def main(root: str, model_path: str, outdir: str, folder: str, feature: str):
    root = pathlib.Path(root)
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 取指定 folder 的测试 loader
    tr, va, te = get_dataloaders(root, batch=1)
    # 只保留 folder 测试集：get_dataloaders 已按序划分，第三 loader 对应 folder=7
    test_loader = te

    # 收集所有窗口预测和真实序列
    model = AutoRegressiveTCN(hidden=128).to('cpu')
    # Load checkpoint, allow missing keys if head layer names differ
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.eval()

    preds, gts = [], []
    for X, y in test_loader:
        y0 = y[:, 0, :]
        with torch.no_grad():
            p = model(X, y0).numpy()[0]  # (HORIZON, D_OUT)
        preds.append(p)
        gts.append(y.numpy()[0])       # (HORIZON, D_OUT)
    pred_windows = np.stack(preds, axis=0)  # (N, H, D)
    gt_windows   = np.stack(gts, axis=0)

    # 计算 total_length: N + WINDOW + 0 (最后 horizon 不足)
    total_length = pred_windows.shape[0] + WINDOW + 1
    # Stitch
    stitched_pred = stitch_windows(pred_windows, total_length)
    stitched_gt   = stitch_windows(gt_windows,   total_length)

    # 特征选择
    if feature.lower() == 'all':
        dims = list(range(D_OUT))
    else:
        idx = int(feature)
        # 同时绘制左右两条腿相同feature
        dims = [idx, idx + (D_OUT//2)]

    # 绘图
    rows = (len(dims) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4*rows))
    axes = axes.flatten()
    time = np.arange(total_length) / FS
    for i, d in enumerate(dims):
        ax = axes[i]
        ax.plot(time, stitched_gt[:, d], label='GT')
        ax.plot(time, stitched_pred[:, d], '--', label='Pred')
        ax.set_title(f'Feature {d}')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Value')
        ax.legend()
    for j in range(len(dims), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    save_path = outdir / f'stitched_feature_{feature}.png'
    plt.savefig(save_path)
    print(f'Saved visualization to {save_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=pathlib.Path(__file__).resolve().parent.parent, help='数据根目录，包含1-7文件夹')
    p.add_argument('--model', default='/media/x/大大747/杂七杂八/25-spring-hw/Sensor/final/IMUdata/runs/model_best.pt', help='训练好的 model_best.pt 路径')
    p.add_argument('--out',     default='viz', help='输出目录')
    p.add_argument('--folder',  default='7', help='测试文件夹编号')
    p.add_argument('--feature', default='all', help='绘制特征索引 0-8 或 all')
    args = p.parse_args()
    main(args.root, args.model, args.out, args.folder, args.feature)
