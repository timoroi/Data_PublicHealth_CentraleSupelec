from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent
input_path = project_root / "data" / "google_trends"
# Charger tes deux fichiers
df_old = pd.read_csv(input_path / "monthly_campaigns_2005_2016.csv")   # 2005–2016
df_new = pd.read_csv(input_path / "google_trends_health_FR_monthly_2015_2026_wide.csv")   # 2015–2026

#pour pas avoir de problème de majuscules
df_old.columns = df_old.columns.str.strip().str.lower()
df_new.columns = df_new.columns.str.strip().str.lower()

df_old["date"] = pd.to_datetime(df_old["date"])
df_new["date"] = pd.to_datetime(df_new["date"])

df_old.set_index("date", inplace=True)
df_new.set_index("date", inplace=True)

# intersection des dates
common = df_old.index.intersection(df_new.index)

df_old_common = df_old.loc[common]
df_new_common = df_new.loc[common]

# dataframe final
result = pd.DataFrame(index=pd.concat([df_old, df_new]).index.unique())

for col in df_old.columns:
    
    # ratio point par point
    ratio = df_new_common[col] / df_old_common[col]
    
    # éviter division par zéro
    ratio = ratio.replace([float('inf'), 0], pd.NA).dropna()
    
    # facteur moyen
    scale = ratio.median()
    
    print(f"{col} → facteur = {scale:.2f}")
    
    # déterminer qui scaler
    if scale > 1:
        df_new[col] = df_new[col] / scale
    else:
        df_old[col] = df_old[col] * scale

# fusion finale
df_final = pd.concat([df_old, df_new]).sort_index()

# enlever doublons (garder new prioritaire)
df_final = df_final[~df_final.index.duplicated(keep="last")]

# arrondir à 1 décimale
df_final = df_final.round(1)

df_final.to_csv("merged_trends.csv")

