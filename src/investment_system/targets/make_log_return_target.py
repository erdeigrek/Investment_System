import pandas as pd
import numpy as np
from investment_system.features.price_features import sort_data

def make_log_return_target(df: pd.DataFrame, symbol_col: str = "symbol", date_col: str = "date", close_col: str = "close", horizon: int = 1) -> pd.DataFrame:
    
    out = df.copy()
    required = {symbol_col, date_col, close_col}
    missing = required - set(out.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    
    out = sort_data(out,[symbol_col,date_col])
    mask = out.duplicated(subset=[symbol_col, date_col])

    if mask.any():
        raise ValueError("Duplicated data exists.")
    
    if not isinstance(horizon, int):
        raise TypeError("Horizon variable  must be an integer type")
    
    if horizon <= 0:
        raise ValueError("Horizon value must be greater than 0")
    
    future_price = out.groupby(symbol_col)[close_col].shift(-horizon)
    
    target_col = f"target_log_ret_{horizon}d"
    if target_col in out.columns:
        raise ValueError(f"Column {target_col} exists. You can't overwrite this column.")
    out[target_col] = np.log(future_price/out[close_col])


    return out