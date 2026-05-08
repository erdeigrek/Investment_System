from investment_system.ingestion.run_market_data import main
from investment_system.pipelines.make_dataset import make_dataset_from_parquet
from investment_system.strategies.baseline import run_baseline_strategy
from backtests.portfolio_backtest import run_portfolio_backtest


DATA_PATH = "C:\\Users\\pawel\\Desktop\\Investment_System\\data\\raw\\prices.parquet"
HORIZON = 20
ROLLING_WINDOWS = (1, 5, 15)
TOP_K = 5


main()
fee_bps = [i for i in range(0, 6)]
sharpe = []
df = make_dataset_from_parquet(DATA_PATH, ROLLING_WINDOWS, HORIZON)
df = run_baseline_strategy(df, TOP_K, HORIZON, ROLLING_WINDOWS)

for i in fee_bps:
    df, portfolio, metrics = run_portfolio_backtest(df, i)
    print(f"\n fee_bps= {i}")
    print(f"Netto Equity = {portfolio['net_equity'].iloc[-1]}")
    print(f"Netto Sharpe = {portfolio['net_sharpe'].iloc[-1]}")
    print(f"Max Drawdown = {metrics['max_drawdown_abs']}")
    print(f"Trade rate = {metrics['trade_rate']}")
    print(len(portfolio))
    sharpe.append(portfolio["net_sharpe"].iloc[-1])

"""
import matplotlib.pyplot as plt
plt.plot(fee_bps,sharpe)
plt.scatter(fee_bps,sharpe)
plt.plot(fee_bps,[0 for i in range(11)],"")
plt.xlabel("Transaction cost (bps)")
plt.ylabel("Sharpe ratio")

plt.title("Strategy performance sensitivity to transaction costs")
plt.show()"""
