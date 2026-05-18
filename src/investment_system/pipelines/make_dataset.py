import pandas as pd
from pathlib import Path
import investment_system.ingestion.market_data as dt
import investment_system.features.price_features as pf


def make_dataset_from_parquet(
    path: Path,
    sentiment_data_root: Path,
    rolling_windows: tuple[int, ...],
    symbol_column: str = "symbol",
    date_column: str = "date",
    close_column: str = "close",
) -> pd.DataFrame:
    df = dt.load_prices(path)
    df = make_dataset(
        df,
        sentiment_data_root,
        rolling_windows,
    )
    return df


def merge_sentiment_and_price(
    df_price: pd.DataFrame, df_sentiment: pd.DataFrame
) -> pd.DataFrame:
    return df_price.merge(df_sentiment, on=("symbol", "date"), how="left").fillna(0)


def make_dataset(
    data: pd.DataFrame,
    data_root: Path,
    rolling_windows: tuple[int, ...],
) -> pd.DataFrame:
    df = data.copy()
    df = pf.add_price_features(df, rolling_windows)
    df_sentiment = pd.read_parquet(data_root)
    df = merge_sentiment_and_price(df, df_sentiment)
    return df


def save_dataset(df: pd.DataFrame, path: Path, horizon: int) -> None:
    full_path = path / f"dataset_h{horizon}.parquet"
    path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(full_path, index=False)
