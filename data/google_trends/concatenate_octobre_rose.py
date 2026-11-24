import pandas as pd
import os

OUTPUT_DIR = "/Users/louis/Desktop/Tronc commun 2A/Projet S8/Data_PublicHealth_CentraleSupelec/data/google_trends"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "octobre_rose_all_seasons.csv")

seasons = [f"{y}_{y + 1}" for y in range(2015, 2026)]

dfs = []
for season in seasons:
    filepath = os.path.join(OUTPUT_DIR, f"octobre_rose_{season}.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, encoding="utf-8-sig")
        dfs.append(df)
        print(f"OK  octobre_rose_{season}.csv  ({len(df)} rows)")
    else:
        print(f"MISSING  octobre_rose_{season}.csv")

combined = pd.concat(dfs, ignore_index=True)
combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"\nTotal : {len(combined)} lignes sauvegardées dans :")
print(f"  {OUTPUT_FILE}")
print("\nAperçu :")
print(combined.head())
