import pandas as pd
import numpy as np
import investment_system.pipelines.make_dataset as md
from investment_system.features.price_features import sort_data
from pathlib import Path

HORIZON = 1
ROLLING_WINDOWS = (1,)
DATA_ROOT = Path(r"data/processed/sentiment_data.parquet")
data = pd.DataFrame(
    {
        "symbol": ["AAPL"] * 6,
        "date": pd.to_datetime(
            [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-06",
            ]
        ),
        "open": [105, 76, 221, 343, 151, 117],
        "close": [100, 102, 101, 105, 107, 110],
    }
)
data = sort_data(data)


def test_make_dataset():
    df = data.copy()
    expected = np.log(df["close"].shift(-1) / df["open"].shift(-1))
    expected = expected.dropna()
    df = md.make_dataset(df, DATA_ROOT, ROLLING_WINDOWS)

    assert "target_log_ret_1d" in df.columns
    assert not df["target_log_ret_1d"].isna().any()
    assert np.allclose(expected, df["target_log_ret_1d"].to_numpy())
    assert len(df) == len(data) - 1
