import os
import time
import random
import pandas as pd
from pytrends.request import TrendReq


# ----------------------------
# Campaign configuration
# ----------------------------
CAMPAIGN_NAME = "Octobre Rose"

CAMPAIGN_QUERY = (
    "Octobre rose + cancer du sein + mammographie "
    "+ dépistage cancer du sein + cancer du sein symptôme"
)

GEO = "FR"

# Fixed window boundaries
WINDOW_START_MONTH, WINDOW_START_DAY = 9, 1    # September 1
WINDOW_END_MONTH,   WINDOW_END_DAY   = 11, 30  # November 30

# ============================================================
# >>> CHANGE THIS to the season you want to fetch <<<
YEAR = 2022
# Example:
# YEAR = 2015  -> fetches 2015-09-01 to 2015-11-30
# and saves with season label 2015_2016
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
    Example: year=2015 -> ('2015-09-01', '2015-11-30')
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
    df["campaign"] = CAMPAIGN_NAME
    df["query"] = CAMPAIGN_QUERY
    df["season"] = f"{year}_{year + 1}"
    df["window_start"] = window_start
    df["window_end"] = window_end
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

    season = f"{YEAR}_{YEAR + 1}"
    window_start, window_end = build_window_dates(YEAR)
    output_path = os.path.join(OUTPUT_DIR, f"octobre_rose_{season}.csv")

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