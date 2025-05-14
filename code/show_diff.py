#!/usr/bin/env python3
"""visualize_diff_tcn.py
================================
用于可视化 TCN+差分模型（AutoRegressiveTCN）在测试集上的连续预测效果，
包括原始预测 vs GT 的每个维度变化曲线，以及整体缝合后的可视化和误差评估。

可选参数：
--feature 0~8  指定单一特征维度（默认 all）
--folder N     指定使用第 N 号文件夹（默认 7）
"""

import argparse, pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from train_ar_tcn import AutoRegressiveTCN, get_dataloaders, D_OUT, WINDOW, HORIZON, FS

def stitch_windows(pred_windows, total_length):
    acc = np.zeros((total_length, D_OUT), dtype=np.float32)
    count = np.zeros((total_length, D_OUT), dtype=np.int32)
    for i in range(pred_windows.shape[0]):
        for h in range(HORIZON):
            t = i + h + 1
            if t < total_length:
                acc[t] += pred_windows[i, h]
                count[t] += 1
    count[count == 0] = 1
    return acc / count

def main(root, model_path, outdir, feature, folder):
    root = pathlib.Path(root)
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model = AutoRegressiveTCN(hidden=128).to('cpu')
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model.eval()

    # 加载数据（仅取 folder 对应的 te）
    tr, va, te = get_dataloaders(root, batch=1)
    loader = te  # 默认 folder 7 为测试

    preds, gts = [], []
    for X, y in loader:
        y0 = y[:, 0, :]
        with torch.no_grad():
            pred = model(X, y0).numpy()[0]
        gt = y.numpy()[0]
        preds.append(pred)
        gts.append(gt)

    pred_windows = np.stack(preds, axis=0)
    gt_windows = np.stack(gts, axis=0)

    total_len = pred_windows.shape[0] + WINDOW + 1
    stitched_pred = stitch_windows(pred_windows, total_len)
    stitched_gt   = stitch_windows(gt_windows, total_len)

    # 绘图维度选择
    if feature == 'all':
        dims = list(range(D_OUT))
    else:
        idx = int(feature)
        dims = [idx, idx + D_OUT//2]  # 左右腿该特征

    rows = (len(dims) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4*rows))
    axes = axes.flatten()
    time = np.arange(total_len) / FS
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
    out_path = outdir / f'diff_tcn_feature_{feature}.png'
    plt.savefig(out_path)
    print(f'[✓] Saved: {out_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=pathlib.Path(__file__).resolve().parent.parent, help='数据根目录，包含1-7文件夹')
    parser.add_argument('--model', default='/media/x/大大747/杂七杂八/25-spring-hw/Sensor/final/IMUdata/runs_diff/model_best.pt', help='训练好的 model_best.pt 路径')
    parser.add_argument('--out', default='viz')
    parser.add_argument('--feature', default='all')
    parser.add_argument('--folder', default='7')
    args = parser.parse_args()
    main(args.root, args.model, args.out, args.feature, args.folder)
