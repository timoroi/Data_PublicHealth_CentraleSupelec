import pandas as pd
import os

OUTPUT_DIR = "/Users/louis/Desktop/Tronc commun 2A/Projet S8/Data_PublicHealth_CentraleSupelec/data/google_trends"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "mars_bleu_all_seasons.csv")

years = range(2015, 2026)

dfs = []
for year in years:
    filepath = os.path.join(OUTPUT_DIR, f"mars_bleu_{year}.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, encoding="utf-8-sig")
        dfs.append(df)
        print(f"OK  mars_bleu_{year}.csv  ({len(df)} rows)")
    else:
        print(f"MISSING  mars_bleu_{year}.csv")

if not dfs:
    print("\nAucun fichier trouvé. Lance d'abord les scripts de détection de peak par année.")
    exit(1)

combined = pd.concat(dfs, ignore_index=True)
combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"\nTotal : {len(combined)} lignes sauvegardées dans :")
print(f"  {OUTPUT_FILE}")
print("\nAperçu :")
print(combined.head())
