import pathlib
import re
from typing import Dict
import pandas as pd
import numpy as np
import io


def load_imu_folder(folder: str, fs: float = 100.0) -> Dict[str, pd.DataFrame]:
    """
    读取“一个数字文件夹”下的全部 IMU txt 文件，返回 {中文放置位置: DataFrame}。
    PacketCounter 会被换算成 time(秒)。

    参数
    ----
    folder : str  数字文件夹路径（内部直接放 txt）
    fs     : float  采样率 Hz，默认 100

    返回
    ----
    Dict[str, pandas.DataFrame]
    """
    
    # 设备 ID 与中文放置位置对照
    ID2NAME = {
        "00B4938D": "左腰",
        "00B492A2": "中间",
        "00B49380": "右腰",
        "00B4938A": "后腰",
        "00B492A5": "左腿",
        "00B492A8": "右腿",
    }

    # 数据列标题
    COLUMNS = [
        "PacketCounter", "Acc_X", "Acc_Y", "Acc_Z",
        "FreeAcc_E", "FreeAcc_N", "FreeAcc_U",
        "Roll", "Pitch", "Yaw",
        "Vel_E", "Vel_N", "Vel_U"
    ]
    
    
    folder_path = pathlib.Path(folder)
    assert folder_path.is_dir(), f"路径不存在或不是文件夹: {folder}"
    
    buckets: Dict[str, list[pd.DataFrame]] = {v: [] for v in ID2NAME.values()}
    id_pattern = re.compile(r"([0-9A-F]{8})", re.IGNORECASE)
    for txt in folder_path.glob("*.txt"):
        parts = txt.stem.split("_")
        if len(parts) < 4:
            print(f"文件名格式错误: {txt.stem}，跳过")
            continue
        dev_id = parts[3].upper()  # 直接获取第三个下划线后的部分并转换为大写
        if dev_id not in ID2NAME:
            print(f"未知设备 ID: {dev_id}，跳过")
            continue
        place = ID2NAME[dev_id]

        # 跳过开头以 // 的注释行和紧随其后的标签行
        with txt.open("r", encoding="utf-8") as f:
            data_lines = []
            skip_next = False
            for line in f:
                if skip_next:
                    skip_next = False
                    continue
                if line.startswith("//"):  # 跳过注释行
                    skip_next = True  # 标记下一行为标签行
                    continue
                if line.startswith("PacketCounter"):  # 跳过注释行
                    skip_next = False   # 标记下一行为标签行
                    continue
                if line.strip():  # 跳过空行
                    data_lines.append(line)

        # 将读取的内容拼接成字符串并解析为 DataFrame
        df = pd.read_csv(
            io.StringIO("".join(data_lines)),
            sep=r"\s+",  # 使用正则表达式匹配空白分隔符
            names=COLUMNS,
            engine="python"
        )
        df["PacketCounter"] = pd.to_numeric(df["PacketCounter"], errors="coerce")
        if df["PacketCounter"].isnull().any():
            raise ValueError(f"文件 {txt.name} 中包含无效的 PacketCounter 数据，无法转换为数值类型。")

        # 替换标签为时间，并对数据进行线性递增处理
        df["PacketCounter"] = range(1, len(df) + 1)  # 将第一个数字变为 1，后续按顺序递增
        df["time"] = df["PacketCounter"] / 100  # 转换为秒
        df.drop(columns="PacketCounter", inplace=True)  # 删除 PacketCounter 列

        # 删除 "Vel_E", "Vel_N", "Vel_U" 列
        columns_to_drop = ["Vel_E", "Vel_N", "Vel_U"]
        df.drop(columns=columns_to_drop, inplace=True, errors="ignore")  # 忽略不存在的列

        buckets[place].append(df)

    # 合并同一位置的所有片段
    out = {}
    for place, dfs in buckets.items():
        if dfs:
            big = pd.concat(dfs, ignore_index=True).sort_values("time")
            big.reset_index(drop=True, inplace=True)
            out[place] = big
    return out


# ---------------- 使用示例 ----------------
if __name__ == "__main__":
    one_folder = 'D:\\杂七杂八\\25-spring-hw\\Sensor\\final\\IMUdata\\1'          # 例如只处理数字文件夹 3
    data_dict = load_imu_folder(one_folder)

    for k, v in data_dict.items():
        print(f"{k}: shape={v.shape}, time[0]={v['time'].iat[0]:.2f}s -> {v['time'].iat[-1]:.2f}s")
        tensor = v.drop(columns="time").to_numpy(dtype=np.float32)  # 直接喂 PyTorch