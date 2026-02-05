import pandas as pd
import numpy as np

def target_1d(close: pd.Series) -> pd.Series:
    """    
    Compute next-day logarithmic return:
    log(close(T+1) / close(T)).
    """
    return np.log(close.shift(-1) / close)

def add_target_1d(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["target_1d"] = df2.groupby("symbol")["close"].transform(target_1d)
    return df2

def add_signal(df: pd.DataFrame)-> pd.DataFrame:
    df2 = df.copy()
    required_cols = set()
    rolling_windows = (5,15)
    for window in rolling_windows:
        required_cols.add(f"px_log_return_mean_{window}")
        required_cols.add(f"px_log_return_volatility_{window}")
    missing = required_cols - set(df2.columns)

    if missing:
        raise KeyError(f"Missing columns: {missing}")
    
    cond = (
        (df2["px_log_return_mean_15"] > 0.01) &
        (df2["px_log_return_mean_5"] > 0) &
        (df2["px_log_return_volatility_15"]  < 2* df2["px_log_return_mean_15"].abs())
    )
    df2["signal"] = np.where(cond, 1, 0)

    return df2

def add_exec_return(df: pd.DataFrame) -> pd.DataFrame:
    
    df2 = df.copy()
    required_cols = {"open","close"}
    missing = required_cols - set(df2.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    df2["exec_return"] = np.log(df2["close"]/df2["open"])
    return df2

def add_position(df: pd.DataFrame) -> pd.DataFrame:
    
    df2 = df.copy()
    required_cols = {"symbol","signal"}
    missing = required_cols - set(df2.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    
    df2["position"] = df2.groupby("symbol")["signal"].shift(1)

    return df2