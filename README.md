# Investment_System

## Target
Targetem modelu jest przyszły logarytmiczny zwrot ceny zamknięcia akcji, liczony jako:

log(close(T+1) / close(T))

## Horyzont predykcji
Horyzont predykcji wynosi 1 dzień (next-day return).

Decyzja inwestycyjna podejmowana jest po zamknięciu sesji dnia T (lub równoważnie: rano dnia T+1 przed otwarciem rynku), na podstawie danych historycznych dostępnych do tego momentu.  
Predykcja dotyczy zwrotu z dnia T do dnia T+1.

Realizacja transakcji następuje w dniu T+1.  
W bazowej wersji systemu przyjmuje się uproszczone założenie wejścia w pozycję po cenie Open(T+1) oraz wyjścia po cenie Close(T+1), bez uwzględnienia kosztów transakcyjnych.

## Jednostka obserwacji
Jednostką obserwacji jest pojedyncza para (symbol, dzień handlowy), odpowiadająca jednemu instrumentowi finansowemu oraz jednemu dniowi obserwacyjnemu w zbiorze danych.

Każda obserwacja jest indeksowana czasowo dniem T i wykorzystywana do predykcji zwrotu z dnia T do dnia T+1.

## Feature engineering i spójność czasowa
Cechy oparte na historii cen (np. logarytmiczne stopy zwrotu, średnie kroczące, zmienność) są konstruowane w sposób zapewniający brak wykorzystania informacji z przyszłości (data leakage).

W szczególności cechy kroczące są obliczane wyłącznie na podstawie danych historycznych sprzed dnia T (poprzez jawne przesunięcie czasowe), tak aby żadna informacja z dnia T+1 ani późniejsza nie była wykorzystywana w procesie predykcji.

Transformacje wymagające dopasowania parametrów (np. skalowanie cech) są dopasowywane wyłącznie na zbiorze treningowym.

## Schemat splitu czasowego
W czasie treningu model nie ma dostępu do danych z okresu testowego.

### TRAIN
- zakres dat: 2016-01-01 → 2020-12-31  
- przeznaczenie: uczenie modelu oraz dopasowanie transformacji danych (np. scalerów)

### TEST
- zakres dat: 2021-01-01 → 2023-12-31  
- przeznaczenie: wyłącznie ewaluacja jakości predykcji oraz strategii inwestycyjnej
