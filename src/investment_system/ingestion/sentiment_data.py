from dotenv import load_dotenv
import os
import pandas as pd
from pathlib import Path
import json


def load_news(path: Path) -> pd.DataFrame:

    df = pd.read_parquet(path)
    return df


def load_symbol_news(data_root: Path, symbol: str) -> pd.DataFrame:
    path = Path(data_root) / symbol
    data = Path(path).glob("*.parquet")
    df = pd.DataFrame()
    for month in data:
        one_month = load_news(month)
        df = pd.concat([df, one_month], ignore_index=True)
    return df


def load_all_symbols_news(data_root: Path) -> pd.DataFrame:
    paths = Path(data_root).iterdir()
    symbols = [path.name for path in paths if path.is_dir()]
    df = pd.DataFrame()
    for symbol in symbols:
        df = pd.concat([df, load_symbol_news(data_root, symbol)], ignore_index=True)
    return df


def _parse_entities(x):
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return []
    return x if isinstance(x, list) else []


def _get_sentiment_for_requested_symbol(entities, requested_symbol):
    for entity in entities:
        if isinstance(entity, dict) and entity.get("symbol") == requested_symbol:
            return entity.get("sentiment_score")
    return None


def extract_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df2 = pd.DataFrame(index=df.index)
    df2["date"] = pd.to_datetime(df["published_at"], errors="coerce").dt.date
    df2["symbol"] = df["requested_symbol"]

    parsed_entities = df["entities_json"].apply(_parse_entities)
    df2["sentiment_score"] = [
        _get_sentiment_for_requested_symbol(entities, requested_symbol)
        for entities, requested_symbol in zip(parsed_entities, df["requested_symbol"])
    ]

    return df2


def get_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby(["symbol", "date"])["sentiment_score"].agg("mean").reset_index()
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_sentiment_pipeline(data_root: Path) -> pd.DataFrame:
    df = load_all_symbols_news(data_root)
    df = extract_sentiment(df)
    df = get_daily_sentiment(df)
    return df


if __name__ == "__main__":
    data_root = Path(r"C:\Users\pawel\Desktop\Investment_System\data\marketaux_monthly")
    df = build_sentiment_pipeline(data_root)
    df.to_parquet(
        Path(
            r"C:\Users\pawel\Desktop\Investment_System\data\processed\sentiment_data.parquet"
        )
    )
