import pandas as pd
import numpy as np
import investment_system.pipelines.make_dataset as md
from investment_system.features.price_features import sort_data
data = pd.DataFrame({
    "symbol": ["AAPL"] * 6,
    "date": pd.to_datetime([
        "2025-01-01",
        "2025-01-02",
        "2025-01-03",
        "2025-01-04",
        "2025-01-05",
        "2025-01-06",
    ]),
    "open": [105, 76, 221, 343, 151, 117],
    "close": [100, 102, 101, 105, 107, 110],
})
data = sort_data(data)

def test_make_dataset():
    df = data.copy()
    expected = np.log(df["close"].shift(-1) / df["close"])
    expected = expected.dropna().values
    df = md.make_dataset(df, (1,), 1)

    assert "target_log_ret_1d" in df.columns
    assert not df["target_log_ret_1d"].isna().any()
    assert np.allclose(expected, df["target_log_ret_1d"].to_numpy())
    assert len(df) == len(data) - 1
