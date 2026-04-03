import pandas as pd
import numpy as np


def rebalance_day(df: pd.DataFrame, holding_period: int) -> pd.DataFrame:
    df = df.sort_values("date")
    mask = (pd.factorize(df["date"])[0]%(holding_period) == 0)
    df["rebalance_day"] = np.where(mask, True, False)
    return df


def ml_rank(df: pd.DataFrame) -> pd.DataFrame:
    df["rank"] = df.groupby("date")["score"].rank(ascending=False)

    return df


def signal_decision(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    mask = (
        (df["rebalance_day"] == True)
            )
    
    mask2= (
        (df["rank"] <= top_k ) &
        (df["score"] > 0)
        )
    
    df["signal_decision"] = np.where(mask, np.where(mask2,1 ,0), np.nan) 

    return df

def add_signal_hold(df: pd.DataFrame) -> pd.DataFrame:
    df["signal_hold"] = df.groupby("symbol")["signal_decision"].ffill().fillna(0)
    return df


def add_position(df: pd.DataFrame) -> pd.DataFrame:
    df["position"] = df.groupby("symbol")["signal_hold"].shift(1).fillna(0)
    return df

def add_weight_hold(df: pd.DataFrame, top_k: int) -> pd.DataFrame:

    mask = (
        (df["rebalance_day"])&
        (df["rank"] <= top_k)&
        (df["score"] > 0)
    )
    mask1 = ((df["rank"] <= top_k)&
        (df)["score"] > 0)
    
    df["target_weight"] = np.where(df["rebalance_day"],np.where(mask1,1,0),np.nan)
    df.loc[mask,"target_weight"] = df["score"]
    sum_score = df.groupby("date")["target_weight"].transform("sum")
    df.loc[mask, "target_weight"] = df.loc[mask, "target_weight"]/sum_score[mask]
    df["weight_hold"] = df["target_weight"]
    df["weight_hold"] = df.groupby("symbol")["target_weight"].ffill().fillna(0)
    df["weight"] = df.groupby("symbol")["weight_hold"].shift().fillna(0)

    return df

def run_ml_strategy(df: pd.DataFrame, holding_period, top_k) -> pd.DataFrame:
    df = df.sort_values(["symbol","date"]).reset_index(drop=True)
    df = rebalance_day(df, holding_period)
    df = ml_rank(df)
    df = signal_decision(df, top_k)
    df = add_signal_hold(df)
    df = add_position(df)
    df = add_weight_hold(df, top_k)

    return df