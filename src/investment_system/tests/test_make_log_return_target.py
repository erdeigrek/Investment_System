import numpy as np
import pandas as pd
from investment_system.targets.make_log_return_target import make_log_return_target

def test_make_log_return_target_happy_path():
    data = {"symbol":["AAPL", "AAPL", "NVDA","AAPL", "NVDA", "NVDA"],
            "date":[np.datetime64('2025-01-26'),np.datetime64('2025-01-24'),
                    np.datetime64('2025-01-25'),np.datetime64('2025-01-25'),
                    np.datetime64('2025-01-24'),np.datetime64('2025-01-26')],
            "close":[110,120,110,150,160,110]}
    df = pd.DataFrame(data = data)
    res = make_log_return_target(df, "symbol","date", "close")
    value = res[(res["symbol"] == "AAPL")&
                (res["date"]==np.datetime64("2025-01-24"))]["target_log_ret_1d"].iloc[0]
    value2 = res[(res["symbol"] == "AAPL")&
                (res["date"]== np.datetime64("2025-01-25"))]["target_log_ret_1d"].iloc[0]
    value3 = res[(res["symbol"] == "AAPL")&
                (res["date"]== np.datetime64("2025-01-26"))]["target_log_ret_1d"].iloc[0]
    value4 = res[(res["symbol"] == "NVDA")&
            (res["date"]== np.datetime64("2025-01-26"))]["target_log_ret_1d"].iloc[0]
    assert np.isclose(value, np.log(150/120))
    assert np.isclose(value2, np.log(110/150))
    assert pd.isna(value3)
    assert pd.isna(value4)

