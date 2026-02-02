import numpy as np
import pandas as pd
from investment_system.targets.make_log_return_target import make_log_return_target
import pytest

def test_overwrite_column():
    data = {"symbol":["AAPL", "AAPL", "NVDA","AAPL", "NVDA", "NVDA"],
            "date":[np.datetime64('2025-01-26'),np.datetime64('2025-01-24'),
                    np.datetime64('2025-01-25'),np.datetime64('2025-01-25'),
                    np.datetime64('2025-01-24'),np.datetime64('2025-01-26')],
            "close":[110,120,110,150,160,110]}
    df = pd.DataFrame(data = data)
    df = make_log_return_target(df, "symbol","date", "close")
    with pytest.raises(ValueError):
        make_log_return_target(df, "symbol","date", "close")