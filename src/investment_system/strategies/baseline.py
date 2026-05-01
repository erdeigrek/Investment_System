import pandas as pd
import numpy as np


def _baseline_windows(windows: tuple[int, ...]) -> tuple[int, int]:
    if len(windows) < 2:
        raise ValueError("Baseline strategy requires at least two rolling windows.")
    return windows[-2], windows[-1]


def add_baseline_score(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (1, 5, 15),
) -> pd.DataFrame:
    fast_window, signal_window = _baseline_windows(windows)
    df["score"] = (
        df[f"px_log_return_mean_{signal_window}"]
        * df[f"px_log_return_mean_{fast_window}"]
        / (df[f"px_log_return_volatility_{signal_window}"] + 1e-12)
    )

    df["rank"] = df.groupby("date")["score"].rank(ascending=False)

    return df


def add_baseline_signal_hold(
    df: pd.DataFrame,
    top_k: int,
    windows: tuple[int, ...] = (1, 5, 15),
    score_threshold: float | None = None,
) -> pd.DataFrame:
    fast_window, signal_window = _baseline_windows(windows)
    ratio_col = f"px_log_return_ratio_{signal_window}"
    if ratio_col in df.columns:
        ratio = df[ratio_col]
    else:
        ratio = df[f"px_log_return_mean_{signal_window}"] / (
            df[f"px_log_return_volatility_{signal_window}"] + 1e-12
        )
    thr = ratio.quantile(0.70) if score_threshold is None else score_threshold
    cond = (
        (df[f"px_log_return_mean_{signal_window}"] > 0)
        & (df[f"px_log_return_mean_{fast_window}"] > 0)
        & (ratio > thr)
        & (df["rank"] <= top_k)
    )

    df["signal_hold"] = np.where(cond, 1, 0)

    return df


def add_baseline_position(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"])
    df["position"] = df.groupby("symbol")["signal_hold"].shift(1).fillna(0)
    return df


def add_equal_weight_from_position(df: pd.DataFrame) -> pd.DataFrame:
    n_active = df.groupby("date")["position"].transform("sum")
    df["weight"] = np.where((df["position"] > 0) & (n_active > 0), 1 / n_active, 0)
    return df


def add_rebalance_signal_hold(
    df: pd.DataFrame,
    top_k: int,
    holding_period: int,
    windows: tuple[int, ...] = (1, 5, 15),
    score_threshold: float | None = None,
) -> pd.DataFrame:
    if holding_period <= 0:
        raise ValueError("holding_period must be greater than 0.")

    fast_window, signal_window = _baseline_windows(windows)
    ratio_col = f"px_log_return_ratio_{signal_window}"
    if ratio_col in df.columns:
        ratio = df[ratio_col]
    else:
        ratio = df[f"px_log_return_mean_{signal_window}"] / (
            df[f"px_log_return_volatility_{signal_window}"] + 1e-12
        )
    thr = ratio.quantile(0.70) if score_threshold is None else score_threshold
    cond = (
        (df[f"px_log_return_mean_{signal_window}"] > 0)
        & (df[f"px_log_return_mean_{fast_window}"] > 0)
        & (ratio > thr)
        & (df["rank"] <= top_k)
    )

    dates = pd.Series(df["date"].drop_duplicates().sort_values().to_numpy())
    rebalance_dates = set(dates.iloc[::holding_period])
    df["rebalance_day"] = df["date"].isin(rebalance_dates)
    df["signal_decision"] = np.where(
        df["rebalance_day"],
        np.where(cond, 1, 0),
        np.nan,
    )
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["signal_hold"] = df.groupby("symbol")["signal_decision"].ffill().fillna(0)
    return df


def run_baseline_strategy(
    df: pd.DataFrame,
    top_k: int,
    holding_period: int = 1,
    windows: tuple[int, ...] = (1, 5, 15),
    score_threshold: float | None = None,
) -> pd.DataFrame:
    df = add_baseline_score(df.copy(), windows)
    df = add_rebalance_signal_hold(df, top_k, holding_period, windows, score_threshold)
    df = add_baseline_position(df)
    df = add_equal_weight_from_position(df)
    return df
