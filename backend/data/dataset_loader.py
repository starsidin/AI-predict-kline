import numpy as np
import pandas as pd
from typing import Tuple

EXPECTED_COLUMNS = [
    "datetime",
    "timestamp",
    "log_ret_scaled",
    "hl_range_scaled",
    "vol_z_scaled",
    "atr_norm_scaled",
    "log_ret_std_scaled",
    "ema_ratio_scaled",
    "rsi_norm_scaled",
    "macd_delta_norm_scaled",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "dom_sin",
    "dom_cos",
    "month_sin",
    "month_cos",
]

FEATURE_COLUMNS = [
    col for col in EXPECTED_COLUMNS if col not in ("datetime", "timestamp")
]
TARGET_COLUMN = "log_ret_scaled"


def load_dual_branch_dataset(
    csv_path: str,
    seq_len: int = 20,
    test_ratio: float = 0.2,
    ma_4h_window: int = 16,
    ma_24h_window: int = 96,
    ma_4h_steps: int = 6,
    ma_24h_steps: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load dual-branch dataset using the real preprocessed features.

    The main branch uses sequential 15m features provided by the
    preprocessing pipeline. The auxiliary branch is built by computing
    rolling averages of the *actual* ``log_ret_scaled`` values over
    4-hour and 24-hour windows, avoiding any synthetic price series.
    """

    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp").reset_index(drop=True)

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"数据集中缺少以下列: {', '.join(missing_cols)}"
        )

    data_main = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    target_series = df[TARGET_COLUMN].astype(np.float32)

    # Auxiliary branch uses real rolling statistics of the log returns
    ma_4h = target_series.rolling(window=ma_4h_window).mean()
    ma_24h = target_series.rolling(window=ma_24h_window).mean()
    ma_total_steps = ma_4h_steps + ma_24h_steps

    future_returns = target_series.shift(-1)

    X_main, X_aux, y = [], [], []
    start_idx = max(
        seq_len - 1,
        ma_4h_window + ma_4h_steps - 2,
        ma_24h_window + ma_24h_steps - 2,
    )

    for i in range(start_idx, len(df) - 1):
        seq_main = data_main[i - seq_len + 1 : i + 1]
        if np.isnan(seq_main).any():
            continue

        seq4 = ma_4h.iloc[i - ma_4h_steps + 1 : i + 1].to_numpy(dtype=np.float32)
        seq24 = ma_24h.iloc[i - ma_24h_steps + 1 : i + 1].to_numpy(dtype=np.float32)
        if (
            seq4.shape[0] != ma_4h_steps
            or seq24.shape[0] != ma_24h_steps
            or np.isnan(seq4).any()
            or np.isnan(seq24).any()
        ):
            continue

        future_ret = future_returns.iloc[i]
        if pd.isna(future_ret):
            continue

        X_main.append(seq_main)
        aux_features = np.concatenate([seq4, seq24]).astype(np.float32)
        X_aux.append(aux_features.reshape(ma_total_steps, 1))
        y.append(int(future_ret > 0))

    X_main = np.array(X_main, dtype=np.float32)
    X_aux = np.array(X_aux, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    split_idx = int(len(X_main) * (1 - test_ratio))
    X_train_main, X_test_main = X_main[:split_idx], X_main[split_idx:]
    X_train_aux, X_test_aux = X_aux[:split_idx], X_aux[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train_main, X_train_aux, y_train, X_test_main, X_test_aux, y_test
