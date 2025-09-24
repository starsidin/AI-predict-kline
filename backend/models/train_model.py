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

from model_trainer import ModelTrainer
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 使用绝对导入
from backend.data.dataset_loader import (
    load_dual_branch_dataset,
    DEFAULT_FUTURE_STEPS,
    DEFAULT_TARGET_THRESHOLD,
)


def load_npy_data(data_dir: Path):
    """加载预先保存的npy数据"""
    try:
        # 确保数据目录存在
        if not data_dir.exists():
            print(f"警告: 数据目录 {data_dir} 不存在，尝试创建...")
            data_dir.mkdir(parents=True, exist_ok=True)
            raise FileNotFoundError(f"数据目录 {data_dir} 已创建，但数据文件不存在。请先准备数据文件。")
            
        # 检查所需的数据文件是否存在
        required_files = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy"]
        missing_files = [f for f in required_files if not (data_dir / f).exists()]
        
        if missing_files:
            raise FileNotFoundError(f"在 {data_dir} 中找不到以下数据文件: {', '.join(missing_files)}")
            
        # 加载数据文件
        print(f"从 {data_dir} 加载数据文件...")
        X_train = np.load(data_dir / "X_train.npy")
        y_train = np.load(data_dir / "y_train.npy")
        X_val = np.load(data_dir / "X_val.npy")
        y_val = np.load(data_dir / "y_val.npy")
        
        print(f"数据加载成功! X_train形状: {X_train.shape}, y_train形状: {y_train.shape}")
        return X_train, y_train, X_val, y_val
        
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        print(f"请确保以下文件存在于 {data_dir} 目录中: X_train.npy, y_train.npy, X_val.npy, y_val.npy")
        raise


def _prepare_classification_targets(*arrays: np.ndarray) -> list[np.ndarray]:
    """将标签转换为一维整数数组，确保适用于分类训练。"""

    processed: list[np.ndarray] = []
    for arr in arrays:
        data = np.asarray(arr)
        data = np.squeeze(data)
        if data.ndim == 0:
            data = data.reshape(1)
        if data.ndim != 1:
            raise ValueError("分类标签必须是一维数组，检测到形状: %s" % (data.shape,))

        if data.dtype == np.bool_:
            data = data.astype(np.int64)
        elif np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.int64)
        elif np.issubdtype(data.dtype, np.floating):
            if not np.all(np.isfinite(data)):
                raise ValueError("标签数组中存在无效的浮点值 (NaN/inf)")
            if not np.all(np.isclose(data, np.round(data))):
                raise ValueError("检测到连续型浮点标签，当前训练脚本仅支持分类任务")
            data = np.round(data).astype(np.int64)
        else:
            raise ValueError(f"不支持的标签数据类型: {data.dtype}")

        processed.append(data)

    return processed


def main():
    parser = argparse.ArgumentParser(description="深度学习模型训练脚本")
    parser.add_argument(
        "--model",
        type=str,
        default="gru_dual",
        choices=["tcn", "gru_dual", "tiny_transformer", "lstm", "gru", "cnn_lstm"],
        help="选择训练的模型类型",
    )
    parser.add_argument("--data", type=str, default="../../data", help="预处理数据所在目录")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument(
        "--future_steps",
        type=int,
        default=DEFAULT_FUTURE_STEPS,
        help="双分支数据集中用于计算未来最高价的K线数量",
    )
    parser.add_argument(
        "--target_threshold",
        type=float,
        default=DEFAULT_TARGET_THRESHOLD,
        help="判定上涨类别的阈值(相对当前收盘价的比例)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data)

    if args.model == "gru_dual":
        # 使用用户指定的数据目录
        csv_path = data_dir / 'processed' / 'btcusdt_15m_features.csv'
        
        # 如果用户指定的路径不存在，尝试使用默认路径
        if not csv_path.exists():
            print(f"警告: 在 {csv_path} 未找到CSV文件，尝试使用默认路径...")
            # 使用项目根目录的绝对路径
            project_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
            csv_path = project_root / 'data' / 'processed' / 'btcusdt_15m_features.csv'
        
        # 检查CSV文件是否存在
        if not csv_path.exists():
            print(f"警告: CSV文件 {csv_path} 不存在!")
            # 尝试创建目录
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            raise FileNotFoundError(f"CSV文件 {csv_path} 不存在。请先准备数据文件。")
            
        print(
            "从 {path} 加载双分支数据集 (future_steps={steps}, target_threshold={threshold:.4f})...".format(
                path=csv_path,
                steps=args.future_steps,
                threshold=args.target_threshold,
            )
        )
        (
            X_train_main,
            X_train_aux,
            y_train,
            X_test_main,
            X_test_aux,
            y_test,
        ) = load_dual_branch_dataset(
            str(csv_path),
            future_steps=args.future_steps,
            target_threshold=args.target_threshold,
        )
        X_train = [X_train_main, X_train_aux]
        X_val = [X_test_main, X_test_aux]
        input_shape = (X_train_main.shape[1:], X_train_aux.shape[1:])
        y_val = y_test
    else:
        X_train, y_train, X_val, y_val = load_npy_data(data_dir)
        input_shape = X_train.shape[1:]

    y_train, y_val = _prepare_classification_targets(y_train, y_val)

    train_unique = np.unique(y_train)
    if train_unique.size < 2:
        raise ValueError("训练集仅包含单一类别，无法执行二分类训练")

    all_unique = np.unique(np.concatenate([y_train, y_val]))
    sorted_labels = np.sort(all_unique)
    if not np.array_equal(sorted_labels, np.arange(sorted_labels.size)):
        y_train = np.searchsorted(sorted_labels, y_train)
        y_val = np.searchsorted(sorted_labels, y_val)
        print(
            "检测到非连续标签 {orig}，已重新映射为从0开始的索引".format(
                orig=sorted_labels.tolist()
            )
        )
        all_unique = np.arange(sorted_labels.size)

    num_classes = int(all_unique.size)
    train_distribution = {
        int(label): int((y_train == label).sum()) for label in all_unique
    }
    val_distribution = {
        int(label): int((y_val == label).sum()) for label in all_unique
    }

    print(
        "训练集标签分布: "
        + ", ".join(f"{label}: {count}" for label, count in train_distribution.items())
    )
    print(
        "验证集标签分布: "
        + ", ".join(f"{label}: {count}" for label, count in val_distribution.items())
    )

    missing_val = [label for label, count in val_distribution.items() if count == 0]
    if missing_val:
        print(
            "警告: 验证集中缺少以下类别的样本: {labels}".format(
                labels=missing_val
            )
        )

    trainer = ModelTrainer()
    model = trainer.create_model_by_name(
        args.model, input_shape=input_shape, num_classes=num_classes
    )
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

    trainer.evaluate_model(model, X_val, y_val)


if __name__ == "__main__":
    main()
