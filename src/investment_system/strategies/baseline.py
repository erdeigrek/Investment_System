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

def add_baseline_signal(df: pd.DataFrame,top_k: int)-> pd.DataFrame:
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

def add_exec_return(df: pd.DataFrame) -> pd.DataFrame:
    df["exec_return"] = np.log(df["close"]/df["open"])
    return df


def add_position(df: pd.DataFrame) -> pd.DataFrame:
    """expects df sorted by symbol and date"""
    
    df["position"] = df.groupby("symbol")["signal_hold"].shift(1).fillna(0)
    return df


def add_n_active(df: pd.DataFrame) -> pd.DataFrame:

    df["n_active"] = df.groupby("date")["position"].transform("sum")
    return df

def add_weight(df: pd.DataFrame) -> pd.DataFrame:

    df["weight"] = np.where(df["n_active"] > 0, df["position"] / df["n_active"], 0)
    return df


