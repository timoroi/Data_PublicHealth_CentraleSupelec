import os
import time
import random
import pandas as pd
from pytrends.request import TrendReq


# ----------------------------
# Campaign configuration
# ----------------------------
campaign_name = "Dry January"

campaign_query = "dry january + mois sans alcool + arreter de boire + janvier sans alcool + arreter alcool"

geo = "FR"


# ----------------------------
# Test window
# ----------------------------
test_start = "2024-12-15"
test_end = "2025-01-31"

days_before_peak = 15
days_after_peak = 30


# ----------------------------
# Output directory
# ----------------------------
output_dir = "/Users/louis/Desktop/Tronc commun 2A/Projet S8/Data_PublicHealth_CentraleSupelec/data/google_trends"
os.makedirs(output_dir, exist_ok=True)


# ----------------------------
# Connect to Google Trends
# ----------------------------
pytrends = TrendReq(hl="fr-FR", tz=60)


def long_pause():
    """
    Pause for a random duration to reduce the risk of rate limiting.
    """
    pause = random.uniform(60, 120)
    print(f"Sleeping {pause:.1f} seconds to avoid rate limit...")
    time.sleep(pause)


# ----------------------------
# Pause before first request
# ----------------------------
print("Waiting before first request...")
long_pause()


# ----------------------------
# Step 1: fetch test window
# ----------------------------
print("Fetching initial window...")

pytrends.build_payload(
    kw_list=[campaign_query],
    timeframe=f"{test_start} {test_end}",
    geo=geo,
    gprop=""
)

long_pause()

df = pytrends.interest_over_time()

if df.empty:
    raise ValueError("No data returned for the initial window.")

if "isPartial" in df.columns:
    df = df.drop(columns=["isPartial"])

df = df.reset_index()
df["date"] = pd.to_datetime(df["date"])


# ----------------------------
# Build campaign index
# ----------------------------
df = df.rename(columns={campaign_query: "campaign_index"})


# ----------------------------
# Step 2: detect peak
# ----------------------------
peak_idx = df["campaign_index"].idxmax()
peak_row = df.loc[peak_idx]

peak_date = peak_row["date"]
peak_value = peak_row["campaign_index"]

print("\nDetected peak:")
print(peak_date, peak_value)


# ----------------------------
# Step 3: build local window
# ----------------------------
local_start = (peak_date - pd.Timedelta(days=days_before_peak)).strftime("%Y-%m-%d")
local_end   = (peak_date + pd.Timedelta(days=days_after_peak)).strftime("%Y-%m-%d")

print("\nLocal window:", local_start, "to", local_end)


# ----------------------------
# Pause before second request
# ----------------------------
long_pause()


# ----------------------------
# Step 4: fetch local window
# ----------------------------
print("Fetching local window...")

pytrends.build_payload(
    kw_list=[campaign_query],
    timeframe=f"{local_start} {local_end}",
    geo=geo,
    gprop=""
)

long_pause()

df_local = pytrends.interest_over_time()

if df_local.empty:
    raise ValueError("No data returned for the local window.")

if "isPartial" in df_local.columns:
    df_local = df_local.drop(columns=["isPartial"])

df_local = df_local.reset_index()
df_local["date"] = pd.to_datetime(df_local["date"])


# ----------------------------
# Rename and add metadata
# ----------------------------
df_local = df_local.rename(columns={campaign_query: "campaign_index"})

df_local["campaign"] = campaign_name
df_local["query"] = campaign_query
df_local["detected_peak_date"] = peak_date.strftime("%Y-%m-%d")
df_local["detected_peak_value"] = peak_value
df_local["window_start"] = local_start
df_local["window_end"] = local_end


# ----------------------------
# Reorder columns
# ----------------------------
df_local = df_local[
    [
        "date",
        "campaign_index",
        "campaign",
        "query",
        "detected_peak_date",
        "detected_peak_value",
        "window_start",
        "window_end"
    ]
]


# ----------------------------
# Export CSV
# ----------------------------
output_path = os.path.join(output_dir, "dry_january_test_peak_window.csv")

df_local.to_csv(output_path, index=False, encoding="utf-8-sig")

print("\nSaved CSV to:")
print(output_path)
print("\nPreview:")
print(df_local.head())