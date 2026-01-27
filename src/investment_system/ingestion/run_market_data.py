from pathlib import Path

from investment_system.common.paths import RAW_DIR
from investment_system.common.config import load_config
from investment_system.ingestion.market_data import fetch_stooq_universe, save_prices

def main() -> None:
    cfg = load_config(Path("configs/base.yaml"))
    df = fetch_stooq_universe(cfg)

    out_path = RAW_DIR / "prices.parquet"
    save_prices(df, out_path)

    print(f"Saved prices to: {out_path} | rows={len(df)}")

if __name__ == "__main__":
    main()
