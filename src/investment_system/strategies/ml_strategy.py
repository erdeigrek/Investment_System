import pandas as pd
import numpy as np


def rebalance_day(df: pd.DataFrame, holding_period: int) -> pd.DataFrame:
    df = df.sort_values("date")
    mask = (pd.factorize(df["date"])[0]%(holding_period) == 0)
    df["rebalance_day"] = np.where(mask, True, False)
    return df

def add_signal_decision(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    df["rank"] = df.groupby("date")["score"].rank(ascending=False)

    mask = (
        (df["rebalance_day"] == True)
            )
    
    mask2= (
        (df["rank"] <= top_k ) &
        (df["score"] > 0)
        )
    mask3 = ()
    df["signal_decision"] = np.where(mask, np.where(mask2,1 ,0), np.nan)
    df["signal_hold"] = df.groupby("symbol")["signal_decision"].ffill().fillna(0)
    df["position"] = df.groupby("symbol")["signal_hold"].shift(1).fillna(0)
    return df

df = pd.DataFrame({
    "date": [
        "2024-01-01", "2024-01-01",
        "2024-01-02", "2024-01-02",
        "2024-01-03", "2024-01-03",
        "2024-01-04", "2024-01-04",
        "2024-01-05", "2024-01-05",
        "2024-01-06", "2024-01-06",
    ],
    "symbol": [
        "A", "B",
        "A", "B",
        "A", "B",
        "A", "B",
        "A", "B",
        "A", "B",
    ],
    "score": [
        0.90, 0.10,   # d1 -> A wygrywa
        0.80, 0.20,   # d2
        0.70, 0.30,   # d3
        0.10, 0.90,   # d4 -> B wygrywa
        0.20, 0.80,   # d5
        0.30, 0.70,   # d6
    ]
})

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
holding_period = 3
top_k = 1
print(df[["date", "symbol", "score"]])
df = rebalance_day(df,holding_period)
df = add_signal_decision(df, top_k)

print(df[["date", "symbol", "rebalance_day"]])
print(df[["date", "symbol", "rank", "signal_decision"]].sort_values(["symbol","date"]))
print(df[["date", "symbol", "signal_hold","position"]].sort_values(["symbol","date"]))