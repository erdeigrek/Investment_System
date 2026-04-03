from investment_system.ingestion.run_market_data import main
from investment_system.pipelines.make_dataset import make_dataset_from_parquet
import yaml
from sklearn.linear_model import LinearRegression, Ridge
from datetime import datetime
from investment_system.strategies.baseline import run_baseline_strategy
from investment_system.strategies.ml_strategy import run_ml_strategy
from backtests.portfolio_backtest import run_portfolio_backtest,final_metrics
import pandas as pd
import numpy as np
import openpyxl
horizon = 20
top_k = 5
fee = 5
#main()
df = make_dataset_from_parquet("/home/erde/Investment_System/data/raw/prices.parquet",[1,5,15], horizon = horizon)



with open('/home/erde/Investment_System/configs/folds.yaml', 'r') as f: 
    data = yaml.load(f, Loader=yaml.SafeLoader)

models = {
    "linear": LinearRegression,
    "ridge": Ridge,
}

my_results = []
for name, Model in models.items():
    print(f'======={name}=======')
    for fold in data["folds"]:
        train_start = pd.to_datetime(fold["train_start"])
        train_end   = pd.to_datetime(fold["train_end"])
        test_start  = pd.to_datetime(fold["test_start"])
        test_end    = pd.to_datetime(fold["test_end"])

        columns = ["log_return",
    "px_log_return_mean_1",
    "px_log_return_volatility_1",
    "px_log_return_mean_5",
    "px_log_return_volatility_5",
    "px_log_return_mean_15",
    "px_log_return_volatility_15",
    f"target_log_ret_{horizon}d"
    ]

        df_test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
        df_train = df[(df["date"] >= train_start) & (df["date"] <= train_end)].copy()

        
        train_data = df_train[columns].dropna().copy()
        test_data = df_test[columns].dropna().copy()

        X_train = train_data.drop(columns=f"target_log_ret_{horizon}d")
        y_train = train_data[f"target_log_ret_{horizon}d"]

        X_test = test_data.drop(columns=f"target_log_ret_{horizon}d")
        y_test = test_data[f"target_log_ret_{horizon}d"]
        model = Model()
        model.fit(X_train,y_train)
        score = model.predict(X_test)
        columns_needed = [
            "date",
            "symbol",
            "open",
            "close",
            "log_return",
            "px_log_return_mean_1",
            "px_log_return_volatility_1",
            "px_log_return_mean_5",
            "px_log_return_volatility_5",
            "px_log_return_mean_15",
            "px_log_return_volatility_15",
            f"target_log_ret_{horizon}d"
        ]
        clean_test_df = df_test[columns_needed].dropna()
        clean_test_df["score"] = score
        clean_test_df = run_ml_strategy(clean_test_df,horizon,top_k)
        df_data, portfolio = run_portfolio_backtest(clean_test_df,fee)
        my_results.append(portfolio)
        metrics = final_metrics(portfolio)
        print(f'{df_data.iloc[0]["date"]} -- {df_data.iloc[-1]["date"]}')
        print(f'Netto Equity = {portfolio["net_equity"].iloc[-1]}')
        print(f'Netto Sharpe = {portfolio["net_sharpe"].iloc[-1]}')
        print(f'Max Drawdown = {metrics["max_drawdown_abs"]}')
        print(f'Trade rate = {metrics["trade_rate"]}\n')
for i in range(4):
    print(my_results[i].iloc[-1]["net_equity"])