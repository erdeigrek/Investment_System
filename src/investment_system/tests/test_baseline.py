import pandas as pd
import numpy as np
import investment_system.strategies.baseline as bs
from investment_system.pipelines.make_dataset import make_dataset
dates = pd.date_range(start="2025-01-01", periods=50)


dates = pd.date_range(start="2025-01-01", periods=25)


df = pd.DataFrame({
    "symbol": ["AAPL"] * 25 + ["NVDA"] * 25,
    "date": list(dates) + list(dates),
    "open": np.concatenate([
        np.round(np.linspace(150, 170, 25) + np.random.normal(0, 1, 25), 2), 
        np.round(np.linspace(400, 500, 25) + np.random.normal(0, 2, 25), 2)  
    ]),
    "close": np.concatenate([
        np.round(np.linspace(151, 171, 25) + np.random.normal(0, 1, 25), 2),
        np.round(np.linspace(402, 505, 25) + np.random.normal(0, 2, 25), 2) 
    ])
})

df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

df = make_dataset(df,[2,5,15],1)
df, df_portfolio = bs.run_baseline_backtest(df,1)

def test_weight_value():
    expected = pd.Series(np.where(df["n_active"] > 0, df["position"] / df["n_active"], 0))
    pd.testing.assert_series_equal(
        df["weight"],
        expected,
        check_names = False,
        obj = "Column weight is not equal 1/n_active"
    )

def test_daily_weight():
    mask = (df["n_active"] > 0)
    
    sum_active = df.loc[mask, "sum_of_daily_weights"]
    assert np.isclose(sum_active, 1.0, atol=1e-6).all()
    
    sum_inactive = df.loc[~mask, "sum_of_daily_weights"]
    assert np.isclose(sum_inactive, 0.0, atol=1e-6).all()

def test_sort_data():
    expected = df.sort_values(["symbol","date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        df,
        expected,
        check_names=False,
        obj = "Rows are not grouped by symbols and sorted by date."
    )

def test_signal_position():

    expected = df.groupby("symbol")["signal"].shift(1).fillna(0)

    pd.testing.assert_series_equal(
        df["position"], 
        expected, 
        check_names=False, 
        obj="Column position is not shifted signal column!"
    )

