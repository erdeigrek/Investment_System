from investment_system.ingestion.run_market_data import main
from investment_system.pipelines.make_dataset import make_dataset_from_parquet
from investment_system.features.price_features import add_price_features
from investment_system.strategies.baseline import run_baseline_backtest, final_metrcis
import matplotlib.pyplot as plt
import numpy as np


main()
fee_bps = [i for i in range(0,6)]
sharpe = []
for i in fee_bps:
    df = make_dataset_from_parquet("/home/erde/Investment_System/data/raw/prices.parquet", [1,5,15], 5)
    df, portfolio = run_baseline_backtest(df, 1, i)
    metrics = final_metrcis(portfolio)
    print(f"\n fee_bps= {i}")
    print(f'Netto Equity = {portfolio["net_equity"].iloc[-1]}')
    print(f'Netto Sharpe = {portfolio["net_sharpe"].iloc[-1]}')
    print(f'Max Drawdown = {metrics["max_drawdown_abs"]}')
    print(f'Trade rate = {metrics["trade_rate"]}')
    print(len(portfolio))
    sharpe.append(portfolio["net_sharpe"].iloc[-1])
"""plt.plot(fee_bps,sharpe)
plt.scatter(fee_bps,sharpe)
plt.plot(fee_bps,[0 for i in range(11)],"")
plt.xlabel("Transaction cost (bps)")
plt.ylabel("Sharpe ratio")

plt.title("Strategy performance sensitivity to transaction costs")
plt.show()"""