import pandas as pd
import numpy as np

def add_baseline_score(df: pd.DataFrame)-> pd.DataFrame:
    df["score"] = (
        df["px_log_return_mean_15"] *
        df["px_log_return_mean_5"] /
        (df["px_log_return_volatility_15"] + 1e-12)
    )
    
    df["rank"] = df.groupby("date")["score"].rank(ascending=False)

    return df

def add_baseline_signal_hold(df: pd.DataFrame,top_k: int)-> pd.DataFrame:
    ratio = df["px_log_return_mean_15"] / (df["px_log_return_volatility_15"] + 1e-12)
    thr = ratio.quantile(0.70)
    cond = (
        (df["px_log_return_mean_15"] > 0) & 
        (df["px_log_return_mean_5"] > 0) &
        (ratio > thr) &
        (df["rank"] <=top_k)
    )
    
    df["signal_hold"] = np.where(cond, 1, 0)


    return df


def add_baseline_position(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"])
    df["position"] = df.groupby("symbol")["signal_hold"].shift(1).fillna(0)
    return df


def add_equal_weight_from_position(df: pd.DataFrame) -> pd.DataFrame:

    df["weight"] = np.where(df["position"] > 0, 1/df.groupby("date")["position"].transform("sum"), 0)
    return df

def run_baseline_strategy(df: pd.DataFrame, top_k) -> pd.DataFrame:
    df = add_baseline_score(df)
    df = add_baseline_signal_hold(df,top_k)
    df = add_baseline_position(df)
    df = add_equal_weight_from_position(df)
    return df