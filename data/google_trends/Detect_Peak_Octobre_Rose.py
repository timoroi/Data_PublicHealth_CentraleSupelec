import os
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
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

YEAR = 2024


# ----------------------------
# Broad test window
# ----------------------------
TEST_START = f"{YEAR}-09-01"
TEST_END   = f"{YEAR}-11-30"

# Local window
DAYS_BEFORE_PEAK = 15
DAYS_AFTER_PEAK = 30


# ----------------------------
# Pause configuration
# ----------------------------
PAUSE_MIN = 60
PAUSE_MAX = 120


# ----------------------------
# Output
# ----------------------------
OUTPUT_DIR = "/Users/louis/Desktop/Tronc commun 2A/Projet S8/Data_PublicHealth_CentraleSupelec/data/google_trends"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------
# Helpers
# ----------------------------
def random_pause():
    pause = random.uniform(PAUSE_MIN, PAUSE_MAX)
    print(f"Sleeping {pause:.0f} seconds...")
    time.sleep(pause)


def fetch_trends(pytrends, query, start, end, geo):
    pytrends.build_payload(
        kw_list=[query],
        timeframe=f"{start} {end}",
        geo=geo,
        gprop=""
    )

    random_pause()

    df = pytrends.interest_over_time()

    if df.empty:
        raise ValueError(f"No data for {start} → {end}")

    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])

    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={query: "campaign_index"})

    return df[["date", "campaign_index"]]


# ----------------------------
# Main
# ----------------------------
def main():
    pytrends = TrendReq(hl="fr-FR", tz=60)

    print(f"\n=== {CAMPAIGN_NAME} ===")

    # ----------------------------
    # 1. Broad window
    # ----------------------------
    df_broad = fetch_trends(
        pytrends,
        CAMPAIGN_QUERY,
        TEST_START,
        TEST_END,
        GEO
    )

    df_broad["window_type"] = "broad"

    # ----------------------------
    # 2. Detect peak
    # ----------------------------
    peak_idx = df_broad["campaign_index"].idxmax()
    peak_date = df_broad.loc[peak_idx, "date"]
    peak_value = df_broad.loc[peak_idx, "campaign_index"]

    print("\nPeak detected:")
    print(peak_date, peak_value)

    # ----------------------------
    # 3. Local window
    # ----------------------------
    local_start = (peak_date - pd.Timedelta(days=DAYS_BEFORE_PEAK)).strftime("%Y-%m-%d")
    local_end   = (peak_date + pd.Timedelta(days=DAYS_AFTER_PEAK)).strftime("%Y-%m-%d")

    df_local = fetch_trends(
        pytrends,
        CAMPAIGN_QUERY,
        local_start,
        local_end,
        GEO
    )

    df_local["window_type"] = "local"

    # ----------------------------
    # 4. Combine data
    # ----------------------------
    df_all = pd.concat([df_broad, df_local], ignore_index=True)

    df_all["campaign"] = CAMPAIGN_NAME
    df_all["query"] = CAMPAIGN_QUERY
    df_all["peak_date"] = peak_date
    df_all["peak_value"] = peak_value

    # ----------------------------
    # 5. Save SINGLE CSV
    # ----------------------------
    csv_path = os.path.join(OUTPUT_DIR, f"octobre_rose_{YEAR}_combined.csv")
    df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # ----------------------------
    # 6. SINGLE plot
    # ----------------------------
    plt.figure(figsize=(12, 6))

    # Broad curve
    plt.plot(
        df_broad["date"],
        df_broad["campaign_index"],
        linestyle="--",
        label="Broad window"
    )

    # Local curve
    plt.plot(
        df_local["date"],
        df_local["campaign_index"],
        label="Local window"
    )

    # Peak
    plt.axvline(peak_date, linestyle=":")
    plt.scatter([peak_date], [peak_value], s=80)

    plt.title(f"{CAMPAIGN_NAME} ({YEAR})")
    plt.xlabel("Date")
    plt.ylabel("Google Trends index")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, f"octobre_rose_{YEAR}_combined.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()

    # ----------------------------
    # Done
    # ----------------------------
    print("\nSaved:")
    print(csv_path)
    print(plot_path)


if __name__ == "__main__":
    main()