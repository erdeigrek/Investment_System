# Monitor nauki — Investment System

## Etap 0 — Fundamenty systemu
- [x] Rozumiem po co pyproject.toml i python -m
- [x] Ingestion Stooq → Parquet działa
- [x] Walidacja OHLC/volume (brak duplikatów, brak NaN, sens OHLC)

## Etap 1 — Dane rynkowe i czas
- [ ] Umiejętnie tłumaczę OHLC na “co się stało w dniu”
- [ ] Rozumiem returns (pct_change) vs log returns
- [ ] Rozumiem rolling window i lookahead bias
- [ ] Potrafię wskazać w kodzie, gdzie może wystąpić leakage

## Etap 2 — Feature engineering (price/volume)
- [ ] Umiejętnie projektuję 5–10 sensownych price features (mom/vol/ma)
- [ ] Zapisuję features do data/processed w Parquet
- [ ] Potrafię opisać każdą kolumnę features (co znaczy, skąd pochodzi)

## Etap 3 — Model ML (baseline)
- [ ] Definiuję target (np. sign(next_day_return))
- [ ] Robię split czasowy (walk-forward / TimeSeriesSplit)
- [ ] Trenuję baseline (logreg / tree) i rozumiem metryki

## Etap 4 — Backtest
- [ ] Zamieniam sygnał → pozycja → PnL bez future leak
- [ ] Dodaję koszty transakcyjne
- [ ] Raport: equity curve, drawdown, Sharpe

## Etap 5 — NLP Sentiment (EN/PL/TR)
- [ ] Pipeline news ingestion
- [ ] Sentiment scoring + agregacja dzienna
- [ ] Łączenie z danymi cenowymi (alignment po dacie)
