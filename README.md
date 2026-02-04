# Investment_System

## Target
#### Targetem modelu jest przyszły logarytmiczny zwrot ceny zamknięcia akcji  liczony jako log(close(T+1)/close(T))

## Horyzont
#### Horyzont predykcji wynosi 1 dzień (next-day return), a decyzja inwestycyjna podejmowana jest na koniec dnia T i realizowana na dzień T+1

## Jednostka obserwacji
#### Jednostką obserwacji jest pojedyncza para (symbol, dzień handlowy), reprezentująca stan rynku na koniec dnia T dla danej spółki

## Schemat splitu czasowego 
#### W czasie treningu model nie ma dostępu do danych z okresu testowego. Wszystkie transformacje danych są dopasowywane wyłącznie na zbiorze treningowym.

#### TRAIN:
 - daty: 2016-01-01 → 2020-12-31
- używany do: uczenia modelu + fit scalerów

#### TEST:
 - daty: 2021-01-01 → 2023-12-31
- używany wyłącznie do ewaluacji strategii
