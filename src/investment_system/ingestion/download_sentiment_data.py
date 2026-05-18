import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone

import yaml
import requests
import pandas as pd


BASE_URL = "https://api.marketaux.com/v1/news/all"
LIMIT_PER_REQUEST = 20  # max 20 artykułów na dzień
CONFIG_PATH = Path(r"C:\Users\pawel\Desktop\Investment_System\configs\base.yaml")
OUT_ROOT = Path("data/marketaux_monthly")


class DailyLimitReached(RuntimeError):
    pass


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def day_windows(start_date: str, end_date: str):
    """
    Zwraca kolejne okna dzienne jako:
    (day_start_dt_utc, next_day_dt_utc)
    """
    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC")

    current = pd.Timestamp(year=start.year, month=start.month, day=start.day, tz="UTC")

    while current <= end:
        next_day = current + pd.Timedelta(days=1)
        yield current.to_pydatetime(), next_day.to_pydatetime()
        current = next_day


def to_api_dt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def month_file_path(out_root: Path, symbol: str, month_start: datetime) -> Path:
    return out_root / symbol / f"{month_start.strftime('%Y-%m')}.parquet"


def empty_result_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "requested_symbol",
            "window_start",
            "window_end",
            "page",
            "uuid",
            "published_at",
            "title",
            "description",
            "snippet",
            "url",
            "source",
            "language",
            "entities_json",
        ]
    )


def normalize_item(
    item: dict, symbol: str, day_start: datetime, day_end: datetime
) -> dict:
    return {
        "requested_symbol": symbol,
        "window_start": day_start.strftime("%Y-%m-%d"),
        "window_end": (day_end - pd.Timedelta(seconds=1)).strftime("%Y-%m-%d"),
        "page": 1,
        "uuid": item.get("uuid"),
        "published_at": item.get("published_at"),
        "title": item.get("title"),
        "description": item.get("description"),
        "snippet": item.get("snippet"),
        "url": item.get("url"),
        "source": item.get("source"),
        "language": item.get("language"),
        "entities_json": json.dumps(item.get("entities") or [], ensure_ascii=False),
    }


def get_json_with_retry(
    session: requests.Session,
    params: dict,
    timeout: int = 30,
    max_retries: int = 5,
) -> dict:
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(BASE_URL, params=params, timeout=timeout)

            if resp.status_code == 402:
                raise DailyLimitReached("Dobity dzienny limit requestów w Marketaux.")

            if resp.status_code == 429:
                sleep_s = min(2**attempt, 60)
                print(f"429 rate limit, sleep {sleep_s}s")
                time.sleep(sleep_s)
                continue

            if 500 <= resp.status_code < 600:
                sleep_s = min(2**attempt, 60)
                print(f"{resp.status_code} server error, sleep {sleep_s}s")
                time.sleep(sleep_s)
                continue

            resp.raise_for_status()
            return resp.json()

        except DailyLimitReached:
            raise

        except requests.RequestException as e:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Request failed after {max_retries} retries: {e}"
                ) from e

            sleep_s = min(2**attempt, 60)
            print(f"Request error: {e}. Retry in {sleep_s}s")
            time.sleep(sleep_s)

    raise RuntimeError("Unexpected retry failure.")


def fetch_symbol_day(
    session: requests.Session,
    symbol: str,
    day_start: datetime,
    day_end: datetime,
    api_token: str,
    language: str = "en",
    group_similar: bool = False,
    must_have_entities: bool = True,
    filter_entities: bool = True,
) -> pd.DataFrame:
    """
    Pobiera maksymalnie 20 artykułów (1 request, page=1) dla jednego dnia.
    """
    params = {
        "api_token": api_token,
        "symbols": symbol,
        "published_after": to_api_dt(day_start),
        "published_before": to_api_dt(day_end),
        "limit": LIMIT_PER_REQUEST,
        "page": 1,
        "language": language,
        "group_similar": str(group_similar).lower(),
        "must_have_entities": str(must_have_entities).lower(),
        "filter_entities": str(filter_entities).lower(),
    }

    data = get_json_with_retry(session, params=params)
    batch = data.get("data", [])

    if not batch:
        return empty_result_df()

    rows = [normalize_item(item, symbol, day_start, day_end) for item in batch]
    df = pd.DataFrame(rows)

    if "uuid" in df.columns and df["uuid"].notna().any():
        df = df.drop_duplicates(subset="uuid")
    elif "url" in df.columns and df["url"].notna().any():
        df = df.drop_duplicates(subset="url")

    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(
            df["published_at"], errors="coerce", utc=True
        )
        df = df.sort_values("published_at").reset_index(drop=True)

    return df


def save_month_parquet(
    df: pd.DataFrame, out_root: Path, symbol: str, month_start: datetime
) -> Path:
    out_path = month_file_path(out_root, symbol, month_start)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def backfill_symbols_monthly(
    symbols: list[str],
    start_date: str,
    end_date: str,
    api_token: str,
    out_root: Path,
    resume: bool = True,
    request_pause: float = 0.7,
) -> pd.DataFrame:
    """
    Iteruje po dniach (1 request = 1 dzień = max 20 artykułów),
    ale zapisuje wyniki miesięcznie do plików .parquet.
    """
    summary_rows = []

    with requests.Session() as session:
        for symbol in symbols:
            print(f"\n=== START {symbol} ===")

            # Grupujemy dni według miesiąca
            # klucz: "YYYY-MM" -> lista DataFrames z kolejnych dni
            month_buffers: dict[str, list[pd.DataFrame]] = {}

            try:
                for day_start, day_end in day_windows(start_date, end_date):
                    month_key = day_start.strftime("%Y-%m")
                    month_start = pd.Timestamp(
                        year=day_start.year, month=day_start.month, day=1, tz="UTC"
                    ).to_pydatetime()
                    out_path = month_file_path(out_root, symbol, month_start)

                    # Jeśli cały miesiąc już zapisany — pomijamy wszystkie dni tego miesiąca
                    if resume and out_path.exists():
                        if month_key not in month_buffers:
                            # oznaczamy jako pominięty tylko raz na miesiąc
                            summary_rows.append(
                                {
                                    "symbol": symbol,
                                    "month": month_key,
                                    "status": "skipped_existing",
                                    "rows": None,
                                    "file": str(out_path),
                                }
                            )
                            month_buffers[month_key] = None  # sentinel: pominięty
                        continue

                    # Inicjalizuj bufor dla nowego miesiąca
                    if month_key not in month_buffers:
                        month_buffers[month_key] = []

                    if month_buffers[month_key] is None:
                        # miesiąc oznaczony jako pominięty
                        continue

                    df_day = fetch_symbol_day(
                        session=session,
                        symbol=symbol,
                        day_start=day_start,
                        day_end=day_end,
                        api_token=api_token,
                        language="en",
                        group_similar=True,
                        must_have_entities=True,
                        filter_entities=True,
                    )

                    if not df_day.empty:
                        month_buffers[month_key].append(df_day)

                    print(
                        f"  {symbol} {day_start.strftime('%Y-%m-%d')} "
                        f"-> {len(df_day)} artykułów"
                    )

                    next_day = day_end
                    is_last_day_of_month = (
                        next_day.month != day_start.month
                        or next_day.year != day_start.year
                    )

                    # Sprawdź też czy to ostatni dzień zakresu end_date
                    is_last_day_of_range = day_start.strftime("%Y-%m-%d") == end_date

                    if is_last_day_of_month or is_last_day_of_range:
                        dfs = month_buffers.get(month_key)
                        if dfs:
                            combined = pd.concat(dfs, ignore_index=True)

                            # Deduplikacja po scaleniu wszystkich dni
                            if (
                                "uuid" in combined.columns
                                and combined["uuid"].notna().any()
                            ):
                                combined = combined.drop_duplicates(subset="uuid")
                            elif (
                                "url" in combined.columns
                                and combined["url"].notna().any()
                            ):
                                combined = combined.drop_duplicates(subset="url")

                            if "published_at" in combined.columns:
                                combined = combined.sort_values(
                                    "published_at"
                                ).reset_index(drop=True)

                            saved_path = save_month_parquet(
                                combined, out_root, symbol, month_start
                            )

                            summary_rows.append(
                                {
                                    "symbol": symbol,
                                    "month": month_key,
                                    "status": "saved",
                                    "rows": len(combined),
                                    "file": str(saved_path),
                                }
                            )

                            print(
                                f"  => Zapisano {month_key}: "
                                f"{len(combined)} artykułów -> {saved_path}"
                            )
                        else:
                            summary_rows.append(
                                {
                                    "symbol": symbol,
                                    "month": month_key,
                                    "status": "empty",
                                    "rows": 0,
                                    "file": str(out_path),
                                }
                            )
                            print(f"  => {month_key}: brak artykułów, pominięto zapis")

                    time.sleep(request_pause)

            except DailyLimitReached:
                print("\nDobity dzienny limit Marketaux. Uruchom ponownie jutro.")
                return pd.DataFrame(summary_rows)

            except Exception as e:
                month_key_err = "unknown"
                print(f"ERROR {symbol}: {e}")
                summary_rows.append(
                    {
                        "symbol": symbol,
                        "month": month_key_err,
                        "status": "error",
                        "rows": None,
                        "file": "",
                        "error": str(e),
                    }
                )

    return pd.DataFrame(summary_rows)


def main():
    config = load_config(CONFIG_PATH)

    api_key = os.getenv("MARKETAUX_API_KEY")
    if not api_key:
        raise ValueError("Brak MARKETAUX_API_KEY w environment variables.")

    start_date = config["dates"]["start"]
    end_date = config["dates"]["end"]
    symbols = config["universe"]["us"]

    print("Config loaded:")
    print(f"start_date = {start_date}")
    print(f"end_date   = {end_date}")
    print(f"symbols    = {symbols}")

    summary = backfill_symbols_monthly(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        api_token=api_key,
        out_root=OUT_ROOT,
        resume=True,
    )

    summary_path = Path("marketaux_backfill_summary_monthly.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nZapisano summary do: {summary_path}")


if __name__ == "__main__":
    main()
