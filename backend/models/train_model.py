#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度学习模型训练脚本
==================

该脚本从 ``CSV`` 文件加载已经归一化的 15 分钟 K 线特征，
在训练前按时间顺序划分训练集与测试集，并在长分支上
实时计算指定周期的均线序列，避免引入未来信息。

用户可通过命令行选择训练 ``TCN``、双分支 ``GRU`` 或 ``Tiny-Transformer`` 等模型。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .model_trainer import ModelTrainer


def build_dataset(
    csv_path: Path,
    seq_len: int,
    ma_hours: int,
    ma_steps: int,
    test_ratio: float,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """根据配置从CSV构建训练与测试数据集"""

    df = pd.read_csv(csv_path)

    # 若无标签列，则根据收盘价方向自动生成二分类标签
    if "label" not in df.columns:
        df["label"] = (df["close"].shift(-1) > df["close"]).astype(int)

    labels = df["label"].values
    feature_cols = [c for c in df.columns if c not in {"label", "timestamp", "datetime"}]
    features = df[feature_cols].values

    # 计算长期分支所需的均线序列
    ma_window = int(ma_hours * 60 / 15)  # 将小时数转换为15分钟的步数
    ma_series = df["close"].rolling(window=ma_window).mean()
    ma_matrix = np.column_stack(
        [ma_series.shift(ma_window * (ma_steps - i - 1)) for i in range(ma_steps)]
    )

    X_short, X_long, y = [], [], []
    for i in range(seq_len, len(df)):
        long_feat = ma_matrix[i]
        if np.isnan(long_feat).any():
            continue  # 跳过前期无法计算均线的样本
        X_short.append(features[i - seq_len : i])
        X_long.append(long_feat)
        y.append(labels[i])

    X_short = np.asarray(X_short)
    X_long = np.asarray(X_long).reshape(-1, ma_steps, 1)
    y = np.asarray(y)

    split_idx = int(len(y) * (1 - test_ratio))
    train = (X_short[:split_idx], X_long[:split_idx], y[:split_idx])
    test = (X_short[split_idx:], X_long[split_idx:], y[split_idx:])
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description="深度学习模型训练脚本")
    parser.add_argument("--model", type=str, default="tcn",
                        choices=["tcn", "gru_dual", "tiny_transformer"],
                        help="选择训练的模型类型")
    parser.add_argument("--csv", type=str, default="data/processed/btcusdt_15m_features.csv",
                        help="预处理特征CSV路径")
    parser.add_argument("--seq_len", type=int, default=96, help="短分支时间步长度")
    parser.add_argument("--ma_hours", type=int, default=4, help="长期分支均线的小时窗口")
    parser.add_argument("--ma_steps", type=int, default=6, help="长期分支均线条数")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="测试集占比，使用末尾样本")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    args = parser.parse_args()

    train, test = build_dataset(
        Path(args.csv), args.seq_len, args.ma_hours, args.ma_steps, args.test_ratio
    )
    Xs_train, Xl_train, y_train = train
    Xs_test, Xl_test, y_test = test

    trainer = ModelTrainer()
    model = trainer.create_model_by_name(
        args.model,
        Xs_train.shape[1:],
        Xl_train.shape[1:],
    )
    if model is None:
        raise RuntimeError("模型创建失败，可能未安装TensorFlow")

    trainer.train_deep_learning_model(
        model,
        args.model,
        [Xs_train, Xl_train],
        y_train,
        [Xs_test, Xl_test],
        y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # 训练完成后在测试集上评估模型性能
    loss, acc = model.evaluate([Xs_test, Xl_test], y_test, verbose=0)
    print(f"测试集准确率: {acc:.4f}")


if __name__ == "__main__":
    main()
