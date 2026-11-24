import os
import glob
import pandas as pd


# ----------------------------
# Directory containing the per-season CSVs
# ----------------------------
INPUT_DIR   = "/Users/louis/Desktop/Tronc commun 2A/Projet S8/Data_PublicHealth_CentraleSupelec/data/google_trends"
OUTPUT_FILE = "dry_january_all_seasons.csv"


def main() -> None:
    # Find all per-season CSVs (e.g. dry_january_2015-2016.csv)
    pattern = os.path.join(INPUT_DIR, "dry_january_????-????.csv")
    files   = sorted(glob.glob(pattern))

    if not files:
        print("No per-season CSV files found. Check INPUT_DIR and filename pattern.")
        return

    print(f"Found {len(files)} file(s):")
    for f in files:
        print(f"  {os.path.basename(f)}")

    frames = [pd.read_csv(f, parse_dates=["date"]) for f in files]
    result = pd.concat(frames, ignore_index=True)

    # Sort chronologically
    result = result.sort_values("date").reset_index(drop=True)

    output_path = os.path.join(INPUT_DIR, OUTPUT_FILE)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n✓ Concatenated {len(result)} total rows into:")
    print(f"  {output_path}")
    print("\nPreview (first 5 rows):")
    print(result.head())
    print("\nSeasons present:")
    print(result["season"].unique())


if __name__ == "__main__":
    main()