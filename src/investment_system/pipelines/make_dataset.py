import pandas as pd
from pathlib import Path
import investment_system.ingestion.market_data as dt
import investment_system.features.price_features as pf
from  investment_system.targets.make_log_return_target import make_log_return_target

def make_dataset_from_parquet(path: Path,rolling_windows:tuple[int,...], horizon: int, symbol_column: str = "symbol", date_column:str = "date", close_column: str = "close" ) -> pd.DataFrame:
    df = dt.load_prices(path)
    df = make_dataset(df,rolling_windows, horizon, symbol_column, date_column, close_column)
    return df

def make_dataset(data:pd.DataFrame, rolling_windows:tuple[int,...], horizon: int, symbol_column: str = "symbol", date_column:str = "date", close_column: str = "close" ) -> pd.DataFrame:
    df = data.copy()
    df = pf.add_price_features(df, rolling_windows)
    df = make_log_return_target(df, symbol_column,date_column,close_column, horizon = horizon )
    target_col = f"target_log_ret_{horizon}d"
    df = df.dropna(subset=[target_col] )
    return df

def save_dataset(df: pd.DataFrame, path: Path, horizon: int) -> None:
    full_path = path / f"dataset_h{horizon}.parquet"
    path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(full_path, index = False)

