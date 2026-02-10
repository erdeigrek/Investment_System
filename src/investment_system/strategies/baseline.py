import pandas as pd
import numpy as np

def add_signal(df: pd.DataFrame)-> pd.DataFrame:
    """
    Generate long-only trading signal.

    Signal = 1 when:
    - medium-term momentum (15d) is strong (> 1%)
    - short-term momentum (5d) is positive
    - volatility is controlled relative to trend strength

    Otherwise signal = 0.
    """

    required_cols = set()
    rolling_windows = (5,15)
    for window in rolling_windows:
        required_cols.add(f"px_log_return_mean_{window}")
        required_cols.add(f"px_log_return_volatility_{window}")
    missing = required_cols - set(df.columns)

    if missing:
        raise KeyError(f"Missing columns: {missing}")
    
    cond = (
        (df["px_log_return_mean_15"] > 0) &
        (df["px_log_return_mean_5"] > 0) &
        (df["px_log_return_volatility_15"]  < 2* df["px_log_return_mean_15"].abs())
    )
    df["signal"] = np.where(cond, 1, 0)

    return df

def add_exec_return(df: pd.DataFrame) -> pd.DataFrame:
    df["exec_return"] = np.log(df["close"]/df["close"].shift(1))
    return df


def add_position(df: pd.DataFrame) -> pd.DataFrame:
    """expects df sorted by symbol and date"""
    
    df["position"] = df.groupby("symbol")["signal"].shift(1).fillna(0)
    return df


def add_n_active(df: pd.DataFrame) -> pd.DataFrame:

    df["n_active"] = df.groupby("date")["position"].transform("sum")
    return df

def add_weight(df: pd.DataFrame) -> pd.DataFrame:

    df["weight"] = np.where(df["n_active"] > 0, df["position"] / df["n_active"], 0)
    return df




def add_portfolio_log_return(df: pd.DataFrame) -> pd.DataFrame:
    
    df["sum_of_daily_weights"] =  df.groupby("date")["weight"].transform("sum")
    df["expected"] = np.where(df["n_active"] > 0, 1, 0)
    diff = abs(df["expected"] - df["sum_of_daily_weights"])
    diff_bool = diff > 1e-9

    if diff_bool.any():
        raise ValueError("Sum of daily weights is not equal 0 nor 1")
    
    df["contribution"] = df["exec_return"]*df["weight"]
    df["portfolio_log_return"] = df.groupby("date")["contribution"].transform("sum")

    return df

def create_portfolio(df: pd.DataFrame) -> pd.DataFrame:

    df = df.drop_duplicates(subset=['date'], keep="first")
    df_portfolio = df.loc[:,["date","portfolio_log_return","n_active"]]

    return df_portfolio.sort_values("date")

def add_portfolio_equity(df: pd.DataFrame, initial_equity: float) -> pd.DataFrame:

    if initial_equity <= 0:
        raise ValueError("initial_equity must be greater than 0.")

    df["cum_log"] = df["portfolio_log_return"].cumsum()
    df["equity"] = initial_equity*np.exp(df["cum_log"])
    return df
    
def add_drawdown(df:pd.DataFrame) -> pd.DataFrame:
    """expects df sorted by date ascending"""

    df["peak"] = df["equity"].cummax()
    df["drawdown"] = df["equity"]/df["peak"] - 1
    return df

def add_expanding_stats(df:pd.DataFrame) -> pd.DataFrame:
    df["expanding_mean"] = df["portfolio_log_return"].expanding().mean().reset_index( drop=True)
    df["expanding_std"]  = df["portfolio_log_return"].expanding().std().reset_index( drop=True)
    mask = (df["expanding_std"] > 0)
    df["Sharpe"] = np.where(mask, (df["expanding_mean"] / df["expanding_std"])*np.sqrt(252),np.nan)
    return df

def final_metrcis(df:pd.DataFrame) -> dict:
    score = {}
    score["mean_daily"] = df["portfolio_log_return"].mean(ddof=0)
    score["std_daily"]  = df["portfolio_log_return"].std(ddof=0)
    if score["std_daily"] != np.nan and score["std_daily"] > 0:
        score["Sharpe"] = (score["mean_daily"]/score["std_daily"])*np.sqrt(252)
    else:
        score["Sharpe"] = np.nan
    score["max_drawdown_abs"] = df["drawdown"].abs().max()
    score["avg_n_actives"] = df["n_active"].median()
    score["trading_days"] = (df["n_active"]>0).sum()
    score["n_days"] = len(df)
    return score
# Baseline Pipeline
def run_baseline_backtest(data: pd.DataFrame, initial_equity) -> tuple[pd.DataFrame, pd.DataFrame]:
    """expects df with price features"""
    df =  data.copy()
    required_cols = {"symbol", "date", "close","open", "px_log_return_mean_15", "px_log_return_mean_5", "px_log_return_volatility_15","px_log_return_volatility_5"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df = add_signal(df)
    df = add_exec_return(df)
    df = add_position(df)
    df = add_n_active(df)
    df = add_weight(df)
    df = add_portfolio_log_return(df)
    df_portfolio = create_portfolio(df)
    df_portfolio = add_portfolio_equity(df_portfolio, initial_equity)
    df_portfolio = add_drawdown(df_portfolio)
    df_portfolio = add_expanding_stats(df_portfolio)

    return df, df_portfolio


