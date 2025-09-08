#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练脚本示例
================

该脚本演示如何从命令行选择不同的深度学习模型进行训练。
默认假设预处理后的特征与标签以 ``npy`` 文件形式保存在 ``data`` 目录下。
用户可通过 ``--model`` 参数选择 TCN、双分支GRU 或 Tiny-Transformer 等模型。
"""

import argparse
from pathlib import Path
import numpy as np

from .model_trainer import ModelTrainer


def load_data(data_dir: Path):
    """加载训练与验证数据"""
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")
    return X_train, y_train, X_val, y_val


def main():
    parser = argparse.ArgumentParser(description="深度学习模型训练脚本")
    parser.add_argument(
        "--model",
        type=str,
        default="tcn",
        choices=["tcn", "gru_dual", "tiny_transformer", "lstm", "gru", "cnn_lstm"],
        help="选择训练的模型类型",
    )
    parser.add_argument("--data", type=str, default="../data", help="预处理数据所在目录")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    args = parser.parse_args()

    data_dir = Path(args.data)
    X_train, y_train, X_val, y_val = load_data(data_dir)

    trainer = ModelTrainer()
    model = trainer.create_model_by_name(args.model, input_shape=X_train.shape[1:])
    if model is None:
        raise RuntimeError("模型创建失败，可能未安装TensorFlow")

    trainer.train_deep_learning_model(
        model,
        args.model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
