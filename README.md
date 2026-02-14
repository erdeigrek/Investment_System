# Investment Decision Support System

System wspomagania decyzji inwestycyjnych oparty na danych rynkowych i metodach Machine Learning.

Projekt realizowany w ramach pracy dyplomowej.

---

## 🎯 Cel projektu

Celem projektu jest:

- zbudowanie kompletnego pipeline’u danych rynkowych,
- implementacja benchmarkowej strategii (baseline),
- zaprojektowanie systemu umożliwiającego porównanie modeli ML z realistycznym punktem odniesienia,
- analiza stabilności i ryzyka strategii.

---

## 🏗 Aktualny stan projektu  
### Milestone 1 – Baseline Complete ✅

### 1️⃣ Pipeline danych
- ingestion danych rynkowych,
- walidacja i sortowanie czasowe,
- feature engineering:
  - log returns,
  - rolling mean,
  - rolling volatility,
- brak look-ahead bias (jawne przesunięcia w rollingach).

---

### 2️⃣ Dataset Builder
- konstrukcja cech + targetu,
- target: przyszły log-return (horyzont predykcji),
- usuwanie obserwacji bez targetu,
- jednoznaczne miejsce powstawania targetu.

---

### 3️⃣ Baseline Backtest
- rule-based signal,
- `position = shift(signal)`,
- execution: intraday (`log(close/open)`),
- równoważony portfel (`1 / n_active`),
- cash day przy braku aktywnych pozycji,
- agregacja portfela do poziomu dziennego,
- equity curve (`exp(cumsum(log_returns))`),
- drawdown,
- metryki końcowe:
  - `mean_daily`
  - `std_daily`
  - `Sharpe` (annualized)
  - `max_drawdown_abs`
  - `median_n_active`
  - `n_days`

---

### 4️⃣ Testy inwariantów
- poprawne przesunięcie sygnału (brak leakage),
- brak mieszania tickerów,
- suma wag = 1 lub 0,
- cash day → brak zwrotu,
- poprawność drawdown.

---

## 🧠 Architektura systemu
**raw prices**
**↓**
**price_features**
**↓**
**make_dataset (features + target)**
**↓**
**baseline strategy**
**↓**
**portfolio aggregation**
**↓**
**metrics**


### Moduły

- `market_data.py` – ingestion i zapis danych
- `price_features.py` – budowa cech cenowych (past-only)
- `make_log_return_target.py` – konstrukcja targetu
- `make_dataset.py` – integracja features + target
- `baseline.py` – benchmarkowa strategia portfelowa

---

## ⏱ Konwencja czasu

- Wiersz danych T reprezentuje stan po close(T)
- Decyzja podejmowana po close(T) / rano T+1
- Sygnał liczony w T-1 steruje pozycją w T
- Execution: open(T) → close(T)
- Brak look-ahead bias

---

## 📊 Metodologia

Strategia benchmarkowa stanowi punkt odniesienia dla przyszłych modeli ML.

Metryki liczone są na poziomie portfela dziennego.

Sharpe ratio annualizowany jest przez √252 (dni handlowe).

---

## 🚧 Następne kroki – Milestone 2

- implementacja kosztów transakcyjnych,
- obliczenie turnover,
- realistyczne ograniczenia portfela,
- diagnostyka stabilności strategii,
- przygotowanie pod walk-forward validation i ML.

---

## ⚠️ Ryzyka projektowe

- data leakage,
- regime shift,
- overfitting przy modelach ML,
- nierealistyczne założenia kosztów,
- nadmierne dopasowanie parametrów baseline.

---

## 📚 Literatura (do uzupełnienia w kolejnych etapach)

Planowane obszary:
- metodologia backtestingu,
- modelowanie kosztów transakcyjnych,
- ocena strategii (Sharpe, drawdown, tail risk),
- walidacja czasowa w ML dla danych finansowych.
