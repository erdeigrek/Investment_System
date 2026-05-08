import pandas as pd
import numpy as np
from investment_system.strategies.baseline import run_baseline_strategy
from investment_system.pipelines.make_dataset import make_dataset

HOLDING = 20
ROLLING_WINDOWS = (1, 5, 15)
TOP_K = 5
PERIODS = 730
dates = pd.date_range(start="2022-01-01", periods=PERIODS)


df = pd.DataFrame(
    {
        "symbol": ["AAPL"] * PERIODS + ["NVDA"] * PERIODS,
        "date": list(dates) + list(dates),
        "open": np.concatenate(
            [
                np.round(
                    np.linspace(150, 300, PERIODS) + np.random.normal(0, 1, PERIODS), 2
                ),
                np.round(
                    np.linspace(400, 600, PERIODS) + np.random.normal(0, 2, PERIODS), 2
                ),
            ]
        ),
        "close": np.concatenate(
            [
                np.round(
                    np.linspace(151, 371, PERIODS) + np.random.normal(0, 1, PERIODS), 2
                ),
                np.round(
                    np.linspace(402, 605, PERIODS) + np.random.normal(0, 2, PERIODS), 2
                ),
            ]
        ),
    }
)

df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

df = make_dataset(df, ROLLING_WINDOWS, HOLDING)
df = run_baseline_strategy(df, 5, HOLDING, ROLLING_WINDOWS)


def test_weight_value():
    n_active = df.groupby("date")["position"].transform("sum")

    expected = pd.Series(np.where(df["position"], df["position"] / n_active, 0))
    pd.testing.assert_series_equal(
        df["weight"],
        expected,
        check_names=False,
        obj="Column weight is not equal 1/n_active",
    )


def test_daily_weight():
    n_active = df.groupby("date")["position"].transform("sum")
    weight_sum = df.groupby("date")["weight"].transform("sum")
    mask = n_active > 0

    assert np.isclose(weight_sum[mask], 1.0, atol=1e-6).all()
    assert np.isclose(weight_sum[~mask], 0.0, atol=1e-6).all()


def test_sort_data():
    expected = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        df,
        expected,
        check_names=False,
        obj="Rows are not grouped by symbols and sorted by date.",
    )


def test_signal_position():

    expected = df.groupby("symbol")["signal_hold"].shift(1).fillna(0)

    pd.testing.assert_series_equal(
        df["position"],
        expected,
        check_names=False,
        obj="Column position is not shifted signal_hold column!",
    )
