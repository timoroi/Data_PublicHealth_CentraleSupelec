import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(r"C:\Users\tedvm\Centrale\PoleProjetS8\Data_PublicHealth_CentraleSupelec\data\google_trends")

files = files = [
    OUTPUT_DIR / "dry_january.csv",
    OUTPUT_DIR / "octobre_rose.csv",
    OUTPUT_DIR / "mars_bleu.csv",
    OUTPUT_DIR / "movember.csv",
]

dfs = []

for file in files:
    df = pd.read_csv(file)
    dfs.append(df)

# concaténer tous les fichiers
df_all = pd.concat(dfs)

# pivot → une colonne par campagne
df_pivot = df_all.pivot(index="date", columns="campaign", values="interest")

# optionnel : trier par date
df_pivot = df_pivot.sort_index()

# sauvegarde
df_pivot.to_csv(OUTPUT_DIR / "monthly_campaigns_2005_2026.csv")

print("✅ Fichier combiné créé")