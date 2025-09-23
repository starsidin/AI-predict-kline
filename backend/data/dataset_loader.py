import numpy as np
import pandas as pd
from typing import Tuple


def load_dual_branch_dataset(
    csv_path: str,
    seq_len: int = 20,
    test_ratio: float = 0.2,
    ma_4h_window: int = 16,
    ma_24h_window: int = 96,
    ma_4h_steps: int = 6,
    ma_24h_steps: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load 15m kline features and construct dual-branch dataset.

    The main branch uses sequential 15m features. The auxiliary branch is
    built on-the-fly by computing 4-hour and 24-hour moving averages and
    collecting the most recent ``ma_4h_steps`` and ``ma_24h_steps`` values
    respectively.

    Args:
        csv_path: Path to ``btcusdt_15m_features.csv``.
        seq_len: Length of the main 15m feature sequence.
        test_ratio: Fraction of samples reserved for the test set (taken
            from the tail of the time series).
        ma_4h_window: Number of 15m candles in a 4‑hour window (default 16).
        ma_24h_window: Number of 15m candles in a 24‑hour window (default 96).
        ma_4h_steps: Number of past 4‑hour averages to keep (default 6 → 1 day).
        ma_24h_steps: Number of past 24‑hour averages to keep (default 5 → 5 days).

    Returns:
        Tuple of training and testing arrays:
        ``(X_train_main, X_train_aux, y_train, X_test_main, X_test_aux, y_test)``
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # 由于data_preprocessor.py已移除close列，我们需要从原始数据中计算或加载close值
    # 这里我们使用log_ret_scaled特征的累积和来近似表示价格变化趋势
    feature_cols = [c for c in df.columns if c not in ["datetime", "timestamp"]]
    data_main = df[feature_cols].values.astype(np.float32)
    
    # 使用log_ret_scaled作为价格变化的指标
    if "log_ret_scaled" in df.columns:
        price_change = df["log_ret_scaled"].values
        # 创建一个起始价格为100的虚拟价格序列
        base_price = 100
        close = np.zeros(len(price_change))
        close[0] = base_price
        for i in range(1, len(price_change)):
            # 使用tanh的反函数来还原实际的对数收益率
            # 注意：这是一个近似值，因为原始数据已经过多次转换
            close[i] = close[i-1] * (1 + price_change[i] * 0.01)  # 缩小变化幅度
    else:
        # 如果没有log_ret_scaled，创建一个虚拟的价格序列
        print("警告：找不到log_ret_scaled列，使用虚拟价格序列")
        close = np.linspace(100, 110, len(df)).astype(np.float32)

    # Moving averages
    ma_4h = pd.Series(close).rolling(window=ma_4h_window).mean()
    ma_24h = pd.Series(close).rolling(window=ma_24h_window).mean()
    ma_total_steps = ma_4h_steps + ma_24h_steps

    X_main, X_aux, y = [], [], []
    start_idx = max(
        seq_len - 1,
        ma_4h_window + ma_4h_steps - 2,
        ma_24h_window + ma_24h_steps - 2,
    )

    for i in range(start_idx, len(df) - 1):
        seq_main = data_main[i - seq_len + 1 : i + 1]
        seq4 = ma_4h.iloc[i - ma_4h_steps + 1 : i + 1].values
        seq24 = ma_24h.iloc[i - ma_24h_steps + 1 : i + 1].values
        if np.isnan(seq4).any() or np.isnan(seq24).any():
            continue
        X_main.append(seq_main)
        X_aux.append(np.concatenate([seq4, seq24]).reshape(ma_total_steps, 1))
        y.append(int(close[i + 1] > close[i]))

    X_main = np.array(X_main, dtype=np.float32)
    X_aux = np.array(X_aux, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    split_idx = int(len(X_main) * (1 - test_ratio))
    X_train_main, X_test_main = X_main[:split_idx], X_main[split_idx:]
    X_train_aux, X_test_aux = X_aux[:split_idx], X_aux[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train_main, X_train_aux, y_train, X_test_main, X_test_aux, y_test
