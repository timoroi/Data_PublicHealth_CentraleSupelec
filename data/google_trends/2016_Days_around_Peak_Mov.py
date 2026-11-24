import os
import time
import random
import pandas as pd
from pytrends.request import TrendReq


# ----------------------------
# Campaign configuration
# ----------------------------
CAMPAIGN_NAME = "Movember"

CAMPAIGN_QUERY = (
    "Movember + cancer prostate + movember prostate + cancer prostate psa + cancer prostate traitement"
)

GEO = "FR"

# Fixed window boundaries (month, day)
WINDOW_START_MONTH, WINDOW_START_DAY = 10, 1  # October 1 of year N
WINDOW_END_MONTH,   WINDOW_END_DAY   =  12,  31  # December 31 of year N

# ============================================================
# >>> CHANGE THIS to the season you want to fetch <<<
YEAR = 2016   # e.g. 2015  →  fetches 2015-10-01 to 2015-12-31
# ============================================================

# ----------------------------
# Pause configuration (seconds)
# ----------------------------
PAUSE_MIN = 60
PAUSE_MAX = 120

# ----------------------------
# Output
# ----------------------------
OUTPUT_DIR = "/Users/louis/Desktop/Tronc commun 2A/Projet S8/Data_PublicHealth_CentraleSupelec/data/google_trends"


# ----------------------------
# Helpers
# ----------------------------

def random_pause() -> None:
    """Sleep for a random duration to reduce the risk of rate limiting."""
    pause = random.uniform(PAUSE_MIN, PAUSE_MAX)
    print(f"  Sleeping {pause:.0f} s...")
    time.sleep(pause)


def build_window_dates(year: int) -> tuple[str, str]:
    """
    Return the (start, end) date strings for a given season.
    Example: year=2016  →  ('2016-10-01', '2016-12-31')
    """
    start = f"{year}-{WINDOW_START_MONTH:02d}-{WINDOW_START_DAY:02d}"
    end   = f"{year}-{WINDOW_END_MONTH:02d}-{WINDOW_END_DAY:02d}"
    return start, end


def fetch_trends(pytrends: TrendReq, window_start: str, window_end: str) -> pd.DataFrame:
    """
    Fetch daily Google Trends data for CAMPAIGN_QUERY over [window_start, window_end].
    """
    pytrends.build_payload(
        kw_list=[CAMPAIGN_QUERY],
        timeframe=f"{window_start} {window_end}",
        geo=GEO,
        gprop=""
    )

    random_pause()

    df = pytrends.interest_over_time()

    if df.empty:
        raise ValueError(f"No data returned for window {window_start} → {window_end}.")

    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])

    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={CAMPAIGN_QUERY: "campaign_index"})

    return df[["date", "campaign_index"]]


def add_metadata(df: pd.DataFrame, year: int,
                 window_start: str, window_end: str) -> pd.DataFrame:
    """Attach campaign metadata columns to a fetched DataFrame."""
    df = df.copy()
    df["campaign"]     = CAMPAIGN_NAME
    df["query"]        = CAMPAIGN_QUERY
    df["season"]       = str(year)
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

    season = str(YEAR)
    window_start, window_end = build_window_dates(YEAR)
    output_path = os.path.join(OUTPUT_DIR, f"movember_{season}.csv")

    print(f"\n[{season}]  {window_start} → {window_end}")

    pytrends = TrendReq(hl="fr-FR", tz=60)

    df = fetch_trends(pytrends, window_start, window_end)
    df = add_metadata(df, YEAR, window_start, window_end)
    df = reorder_columns(df)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n✓ {len(df)} rows saved to:")
    print(f"  {output_path}")
    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()