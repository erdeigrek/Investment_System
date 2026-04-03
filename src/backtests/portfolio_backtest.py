import pandas as pd
import numpy as np

def add_exec_return(df: pd.DataFrame) -> pd.DataFrame:
    df["exec_return"] = np.log(df["close"]/df["open"])
    return df



def add_n_active(df: pd.DataFrame) -> pd.DataFrame:

    df["n_active"] = ((df["weight"] > 0).groupby(df["date"]).transform("sum"))
    return df



def add_turnover(df: pd.DataFrame) -> pd.DataFrame:
    df["prev_weight"] = df.groupby("symbol")["weight"].shift().fillna(0)
    df["daily_weight_diff"] = np.abs(df["weight"] - df["prev_weight"])
    df["turnover"] = df.groupby("date")["daily_weight_diff"].transform("sum")

    return df



def add_cost(df:pd.DataFrame, fee_bps:int ) -> pd.DataFrame:
    fee =  fee_bps/10_000
    df["cost"] = df["turnover"]*fee

    return df



def add_gross_log_return(df: pd.DataFrame) -> pd.DataFrame:
    df["contribution"] = df["exec_return"]*df["weight"]
    df["gross_log_return"] = df.groupby("date")["contribution"].transform("sum")

    return df



def create_portfolio(df: pd.DataFrame) -> pd.DataFrame:

    df = df.drop_duplicates(subset=['date'], keep="first")
    df_portfolio = df.loc[:,["date", "gross_log_return", "n_active", "cost", "turnover"]]

    return df_portfolio.sort_values("date")



def add_portfolio_equity(df: pd.DataFrame, initial_equity: float = 1.0) -> pd.DataFrame:

    if initial_equity <= 0:
        raise ValueError("initial_equity must be greater than 0.")

    df["cum_log"] = df["gross_log_return"].cumsum()
    df["equity"] = initial_equity*np.exp(df["cum_log"])
    return df



def add_netto_values(df:pd.DataFrame, initial_equity: float = 1.0) -> pd.DataFrame:
    if (df["cost"] > 1).any():
        raise ValueError("Cost cannot be greater than 1!")
    df["net_log_return"] = df["gross_log_return"] + np.log(1-df["cost"])
    df["net_cum_log"] = df["net_log_return"].cumsum()
    df["net_equity"] = initial_equity*np.exp(df["net_cum_log"])

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




def final_metrics(df:pd.DataFrame) -> dict:
    score = {}
    score["mean_daily"] = df["gross_log_return"].mean()
    score["std_daily"]  = df["gross_log_return"].std(ddof=0)
    if score["std_daily"] > 0:
        score["gross_sharpe"] = (score["mean_daily"]/score["std_daily"])*np.sqrt(252)
    else:
        score["gross_sharpe"] = np.nan
    score["max_drawdown_abs"] = df["drawdown"].abs().max()
    score["trading_days"] = (df["turnover"]>0).sum()
    score["n_days"] = len(df)
    score["trade_rate"] = score["trading_days"]/len(df)
    score["final_equity"] = df.iloc[-1]["equity"]
    score["final_net_equity"] = df.iloc[-1]["net_equity"]
    return score

def run_portfolio_backtest(df: pd.DataFrame, fee_bps: int)-> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df = add_exec_return(df)
    df = add_n_active(df)
    df = add_turnover(df)
    df = add_cost(df, fee_bps)
    df = add_gross_log_return(df)
    portfolio = create_portfolio(df)
    portfolio = add_portfolio_equity(portfolio, 1.0)
    portfolio = add_netto_values(portfolio, 1.0)
    portfolio = add_drawdown(portfolio)
    portfolio = add_expanding_stats(portfolio)
    
    return  df, portfolio