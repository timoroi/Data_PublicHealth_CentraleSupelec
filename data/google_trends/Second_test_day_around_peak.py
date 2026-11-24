import os
import time
import random
import pandas as pd
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError


# ----------------------------
# Campaign configuration
# ----------------------------
CAMPAIGN_NAME = "Dry January"

CAMPAIGN_QUERY = (
    "dry january + mois sans alcool + arreter de boire "
    "+ janvier sans alcool + arreter alcool"
)

GEO = "FR"

# Fixed window boundaries (month, day)
WINDOW_START_MONTH, WINDOW_START_DAY = 12, 26  # December 26 of year N
WINDOW_END_MONTH,   WINDOW_END_DAY   =  2,  6  # February  6 of year N+1

# Years to iterate: 2015 → 2025 (i.e. seasons 2015-2016 … 2025-2026)
FIRST_YEAR = 2015
LAST_YEAR  = 2020

# ----------------------------
# Pause configuration (seconds)
# ----------------------------
# Normal pause between requests / seasons
PAUSE_MIN = 180
PAUSE_MAX = 240

# Exponential backoff on 429
BACKOFF_BASE = 300   # 5 minutes for first retry
BACKOFF_MAX  = 1800  # 30 minutes cap
MAX_RETRIES  = 5

# ----------------------------
# Output
# ----------------------------
OUTPUT_DIR  = "/Users/louis/Desktop/Tronc commun 2A/Projet S8/Data_PublicHealth_CentraleSupelec/data/google_trends"
OUTPUT_FILE = "dry_january_fixed_windows_2015_2021.csv"


# ----------------------------
# Helpers
# ----------------------------

def random_pause(min_sec: int = PAUSE_MIN, max_sec: int = PAUSE_MAX) -> None:
    """Sleep for a random duration to reduce the risk of rate limiting."""
    pause = random.uniform(min_sec, max_sec)
    print(f"  Sleeping {pause:.0f} s...")
    time.sleep(pause)


def backoff_pause(attempt: int) -> None:
    """Exponential backoff after a 429 error, capped at BACKOFF_MAX."""
    wait = min(BACKOFF_BASE * (2 ** attempt), BACKOFF_MAX)
    jitter = random.uniform(0, 30)
    total = wait + jitter
    print(f"  429 received. Backoff attempt {attempt + 1}/{MAX_RETRIES}: "
          f"waiting {total:.0f} s before retry...")
    time.sleep(total)


def build_window_dates(year: int) -> tuple[str, str]:
    """
    Return the (start, end) date strings for a given season.
    Example: year=2015  →  ('2015-12-26', '2016-02-06')
    """
    start = f"{year}-{WINDOW_START_MONTH:02d}-{WINDOW_START_DAY:02d}"
    end   = f"{year + 1}-{WINDOW_END_MONTH:02d}-{WINDOW_END_DAY:02d}"
    return start, end


def fetch_trends(pytrends: TrendReq, window_start: str, window_end: str) -> pd.DataFrame:
    """
    Fetch daily Google Trends data for CAMPAIGN_QUERY over [window_start, window_end].
    Retries up to MAX_RETRIES times on 429 with exponential backoff.

    Returns a cleaned DataFrame with columns [date, campaign_index],
    or raises an exception if all retries are exhausted.
    """
    for attempt in range(MAX_RETRIES):
        try:
            pytrends.build_payload(
                kw_list=[CAMPAIGN_QUERY],
                timeframe=f"{window_start} {window_end}",
                geo=GEO,
                gprop=""
            )

            # Pause after build_payload, before the actual data fetch
            random_pause()

            df = pytrends.interest_over_time()

            if df.empty:
                raise ValueError(
                    f"No data returned for window {window_start} → {window_end}."
                )

            # Drop the partial-week flag added by pytrends
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])

            df = df.reset_index()
            df["date"] = pd.to_datetime(df["date"])
            df = df.rename(columns={CAMPAIGN_QUERY: "campaign_index"})

            return df[["date", "campaign_index"]]

        except TooManyRequestsError:
            if attempt < MAX_RETRIES - 1:
                backoff_pause(attempt)
            else:
                raise RuntimeError(
                    f"Still rate-limited after {MAX_RETRIES} attempts "
                    f"for window {window_start} → {window_end}. Giving up."
                )


def add_metadata(df: pd.DataFrame, year: int,
                 window_start: str, window_end: str) -> pd.DataFrame:
    """Attach campaign metadata columns to a fetched DataFrame."""
    df = df.copy()
    df["campaign"]     = CAMPAIGN_NAME
    df["query"]        = CAMPAIGN_QUERY
    df["season"]       = f"{year}-{year + 1}"
    df["window_start"] = window_start
    df["window_end"]   = window_end
    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce the final column order."""
    return df[[
        "date",
        "campaign_index",
        "campaign",
        "query",
        "season",
        "window_start",
        "window_end",
    ]]


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pytrends = TrendReq(hl="fr-FR", tz=60)

    all_frames: list[pd.DataFrame] = []

    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        season = f"{year}-{year + 1}"
        window_start, window_end = build_window_dates(year)

        print(f"\n[{season}]  {window_start} → {window_end}")

        try:
            df = fetch_trends(pytrends, window_start, window_end)
            df = add_metadata(df, year, window_start, window_end)
            df = reorder_columns(df)

            all_frames.append(df)
            print(f"  ✓ {len(df)} rows fetched.")

        except (ValueError, RuntimeError) as err:
            print(f"  ✗ Skipping {season}: {err}")

        # Inter-season pause (skip after the last year)
        if year < LAST_YEAR:
            print("  Inter-season pause...")
            random_pause()

    if not all_frames:
        print("\nNo data collected. Exiting.")
        return

    result = pd.concat(all_frames, ignore_index=True)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n✓ Saved {len(result)} total rows to:")
    print(f"  {output_path}")
    print("\nPreview (first 5 rows):")
    print(result.head())


if __name__ == "__main__":
    main()