from investment_system.targets.make_log_return_target import make_log_return_target
import pandas as pd
import numpy as np
import pytest

def test_make_log_return_target_raises_on_missing_column():
    data = {"symbol":["AAPL", "AAPL", "NVDA","AAPL", "NVDA", "NVDA"],
            "date":[np.datetime64('2025-01-26'),np.datetime64('2025-01-24'),
                    np.datetime64('2025-01-25'),np.datetime64('2025-01-25'),
                    np.datetime64('2025-01-24'),np.datetime64('2025-01-26')],
            "target_log_ret_1d":[110,120,110,150,160,110]}
    df = pd.DataFrame(data = data)
    with pytest.raises(KeyError):
        make_log_return_target(df,"symbol", "date", "close")