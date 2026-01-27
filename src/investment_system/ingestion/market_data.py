import pandas as pd
from datetime import date
from pathlib import Path

def stooq_symbol(symbol: str, market: str) -> str:
    symbol = symbol.lower()
    if market == "us":
        return f"{symbol}.us"
    if market == "pl":
        return f"{symbol}.wa"
    raise ValueError(f"Unknown market: {market}")

def to_stooq_date(d: date) -> str:
    return d.strftime("%Y%m%d")


def fetch_stooq_symbol(
        symbol: str,
        market: str,
        start: date,
        end: date,) -> pd.DataFrame:
    
    stooq_sym = stooq_symbol(symbol, market)

    url = "https://stooq.com/q/d/l/"
    params = {
        "s": stooq_sym,
        "i": "d",
        "d1": to_stooq_date(start),
        "d2": to_stooq_date(end),
    }

    df = pd.read_csv(url, params=params)

    if df.empty:
        raise ValueError(f"No data returned for {stooq_sym}")

    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = symbol.upper()
    df["market"] = market

    return df.sort_values("date").reset_index(drop=True)

def validate_prices(df: pd.DataFrame) -> None:
    required_columns = ["date", "open", "high", "low", "close", "volume", "symbol", "market"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if df[["date", "open", "close"]].isnull().any().any():
        raise ValueError("Price columns contain null values")

    if (df["high"] < df["low"]).any():
        raise ValueError("High price is less than low price for some records")

    if (df["open"] <= 0).any() or (df["high"] <= 0).any() or (df["low"] <= 0).any() or (df["close"] <= 0).any():
        raise ValueError("Price columns contain negative values or equals zero")

    if (df["volume"] < 0).any():
        raise ValueError("Volume column contains negative values")
    
    if (df["high"] < df[["open", "close"]].max(axis=1)).any():
        raise ValueError("High is lower than open/close for some rows")

    if (df["low"] > df[["open", "close"]].min(axis=1)).any():
        raise ValueError("Low is higher than open/close for some rows")
    
    if df.duplicated(["date", "symbol", "market"]).any():
        raise ValueError("Duplicate (market, symbol, date) rows detected")    


def fetch_stooq_universe(cfg: dict) -> pd.DataFrame:
    start = date.fromisoformat(cfg["dates"]["start"])
    end = date.fromisoformat(cfg["dates"]["end"])

    frames = []

    for symbol in cfg["universe"]["us"]:
        frames.append(fetch_stooq_symbol(symbol, "us", start, end))

    for symbol in cfg["universe"]["pl"]:
        frames.append(fetch_stooq_symbol(symbol, "pl", start, end))

    df = pd.concat(frames, ignore_index=True)
    validate_prices(df)

    return df


def save_prices(df: pd.DataFrame, path: Path) -> None:
    """Saves prices DataFrame to Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)



