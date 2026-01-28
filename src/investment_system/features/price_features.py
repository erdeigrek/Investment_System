import pandas as pd
import numpy as np
import pathlib as Path
def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)

def log_return(close: pd.Series) -> pd.Series:
    """Log Return - """
    return np.log(close / close.shift(1))

def add_log_return(df: pd.DataFrame) -> pd.DataFrame:

    df2 = df.copy()
    df2["log_return"] = (
    df2.groupby("symbol")["close"].transform(log_return))
    return df2

def px_log_return_mean(log_ret: pd.Series,window: int,min_periods: int |  None = None) -> pd.Series:
    """
    Rolling mean of log returns based on past observations only.
    The feature is shifted by one period to avoid look-ahead bias.
    """
    if min_periods is None:
        min_periods = window
    return log_ret.shift(1).rolling(window = window, min_periods= min_periods).mean()

    
def add_px_log_return_mean(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Create column with values of Rolling mean of log returns"""
    df2 = df.copy()
    df2["px_log_return_mean_"+ str(window)] = (
    df2.groupby("symbol")["log_return"].transform(lambda s:px_log_return_mean(s, window)))
    return df2

def px_log_return_volatility(log_ret: pd.Series,window: int,min_periods: int |  None = None) -> pd.Series:
    """
    Rolling volatility of log returns based on past observations only.
    The feature is shifted by one period to avoid look-ahead bias.
    """
    if min_periods is None:
        min_periods = window
    return log_ret.shift(1).rolling(window = window, min_periods= min_periods).std(ddof=0)

    
def add_px_log_return_volatility(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df2 = df.copy()
    df2["px_log_return_volatility_"+ str(window)] = (
    df2.groupby("symbol")["log_return"].transform(lambda s:px_log_return_volatility(s, window)))
    return df2


def validate_data(df: pd.DataFrame) -> None:

    required_cols = {"symbol", "date", "close","open"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise TypeError("Column 'date' must be datetime64[ns]")

    if not pd.api.types.is_string_dtype(df["symbol"]):
        raise TypeError("Column symbol must be a string")
    if not pd.api.types.is_numeric_dtype(df["open"]) :
        raise TypeError("Column Open must be a numeric")
    if not pd.api.types.is_numeric_dtype(df["close"]) :
        raise TypeError("Column Close must be a numeric")
    if (df["close"] <= 0).any():
        raise ValueError("Column 'close' must contain only positive values")

def add_price_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    validate_data(df)
    df = sort_data(df)
    df = add_log_return(df)
    df = add_px_log_return_mean(df,(5,10,15))
    df = add_px_log_return_volatility(df,(5,10,15))

    return df

