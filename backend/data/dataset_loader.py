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

    feature_cols = [c for c in df.columns if c not in ["datetime"]]
    data_main = df[feature_cols].values.astype(np.float32)
    close = df["close"].values.astype(np.float32)

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
