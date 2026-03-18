import pandas as pd
import numpy as np

def add_signal(df: pd.DataFrame,top_k: int)-> pd.DataFrame:
    """
    Generate long-only trading signal.

    Signal = 1 when:
    - medium-term momentum (15d) is strong (> 0%)
    - short-term momentum (5d) is positive
    - volatility is controlled relative to trend strength

    Otherwise signal = 0.
    """
    if "signal" in df.columns and df["signal"].notna().any():
        return df
    
    ratio = df["px_log_return_mean_15"] / (df["px_log_return_volatility_15"] + 1e-12)
    thr = ratio.quantile(0.70)
    df["score"] = (
        df["px_log_return_mean_15"] *
        df["px_log_return_mean_5"] /
        (df["px_log_return_volatility_15"] + 1e-12)
    )
    df["rank"] = df.groupby("date")["score"].rank(ascending=False)
    cond = (
        (df["px_log_return_mean_15"] > 0) &
        (df["px_log_return_mean_5"] > 0) &
        (ratio > thr) &
        (df["rank"] <=top_k)
    )
    
    df["signal"] = np.where(cond, 1, 0)


    return df

def add_exec_return(df: pd.DataFrame) -> pd.DataFrame:
    df["exec_return"] = np.log(df["close"]/df["open"])
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




def add_gross_log_return(df: pd.DataFrame,fee_bps: int) -> pd.DataFrame:
    
    df["sum_of_daily_weights"] =  df.groupby("date")["weight"].transform("sum")
    if df["sum_of_daily_weights"].isna().any():
        raise ValueError("Sum of daily weights cannot be NaN")
    df["turnover"] = np.where(df["n_active"] > 0, 2.0, 0.0)
        
    fee =  fee_bps/10_000
    df["cost"] = df["turnover"]*fee
    if df["cost"].min() < 0:
        raise ValueError("Cost cannot be lower than 0")
    if df["cost"].max() >= 1:
        raise ValueError("Cost cannot be greater or equal 1")
    

    df["expected"] = np.where(df["n_active"] > 0, 1, 0)
    diff = abs(df["expected"] - df["sum_of_daily_weights"])
    diff_bool = diff > 1e-9

    if diff_bool.any():
        raise ValueError("Sum of daily weights is not equal 0 nor 1")
    
    df["contribution"] = df["exec_return"]*df["weight"]
    df["gross_log_return"] = df.groupby("date")["contribution"].transform("sum")

    return df

def create_portfolio(df: pd.DataFrame) -> pd.DataFrame:

    df = df.drop_duplicates(subset=['date'], keep="first")
    df_portfolio = df.loc[:,["date","gross_log_return","n_active","cost"]]

    return df_portfolio.sort_values("date")

def add_portfolio_equity(df: pd.DataFrame, initial_equity: float) -> pd.DataFrame:

    if initial_equity <= 0:
        raise ValueError("initial_equity must be greater than 0.")

    df["cum_log"] = df["gross_log_return"].cumsum()
    df["equity"] = initial_equity*np.exp(df["cum_log"])
    return df
    
def add_drawdown(df:pd.DataFrame) -> pd.DataFrame:
    """expects df sorted by date ascending"""

    df["peak"] = df["equity"].cummax()
    df["net_peak"] = df["net_equity"].cummax()

    df["drawdown"] = df["equity"]/df["peak"] - 1
    df["net_drawdown"] = df["net_equity"]/df["net_peak"] - 1

    return df

def add_expanding_stats(df:pd.DataFrame) -> pd.DataFrame:
    df["expanding_mean"] = df["gross_log_return"].expanding(min_periods=60).mean()
    df["net_expanding_mean"] = df["net_log_return"].expanding(min_periods=60).mean()

    df["expanding_std"]  = df["gross_log_return"].expanding(min_periods=60).std(ddof = 0)
    df["net_expanding_std"]  = df["net_log_return"].expanding(min_periods=60).std(ddof = 0)

    mask = (df["expanding_std"] > 0)
    df["gross_sharpe"] = np.where(mask, (df["expanding_mean"] / df["expanding_std"])*np.sqrt(252),np.nan)
    net_mask = (df["net_expanding_std"] > 0)
    df["net_sharpe"] = np.where(net_mask, (df["net_expanding_mean"] / df["net_expanding_std"])*np.sqrt(252),np.nan)

    return df

def final_metrcis(df:pd.DataFrame) -> dict:
    score = {}
    score["mean_daily"] = df["gross_log_return"].mean()
    score["std_daily"]  = df["gross_log_return"].std(ddof=0)
    if score["std_daily"] > 0:
        score["gross_sharpe"] = (score["mean_daily"]/score["std_daily"])*np.sqrt(252)
    else:
        score["gross_sharpe"] = np.nan
    score["max_drawdown_abs"] = df["drawdown"].abs().max()
    score["trading_days"] = (df["n_active"]>0).sum()
    score["n_days"] = len(df)
    score["trade_rate"] = score["trading_days"]/len(df)
    return score

def add_netto_values(df:pd.DataFrame, initial_equity) -> pd.DataFrame:

    if initial_equity <= 0:
        raise ValueError("initial_equity must be greater than 0.")
    
    df["net_log_return"] = df["gross_log_return"] + np.log(1-df["cost"])
    df["net_cum_log"] = df["net_log_return"].cumsum()
    df["net_equity"] = initial_equity*np.exp(df["net_cum_log"])

    return df

# Baseline Pipeline
def run_baseline_backtest(data: pd.DataFrame, initial_equity: int, fee_bps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """expects df with price features"""
    df =  data.copy()
    required_cols = {"symbol", "date", "close","open", "px_log_return_mean_15", "px_log_return_mean_5", "px_log_return_volatility_15","px_log_return_volatility_5"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df = add_signal(df,3)
    df = add_exec_return(df)
    df = add_position(df)
    df = add_n_active(df)
    df = add_weight(df)
    df = add_gross_log_return(df, fee_bps)
    df_portfolio = create_portfolio(df)
    df_portfolio = add_portfolio_equity(df_portfolio, initial_equity)
    df_portfolio = add_netto_values(df_portfolio,initial_equity)
    df_portfolio = add_drawdown(df_portfolio)
    df_portfolio = add_expanding_stats(df_portfolio)

    return df, df_portfolio


