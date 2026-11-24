import pandas as pd
import matplotlib.pyplot as plt

# Path to the CSV file
csv_path = "/Users/louis/Desktop/Tronc commun 2A/Projet S8/Data_PublicHealth_CentraleSupelec/data/google_trends/dry_january_test_peak_window.csv"

# Load data
df = pd.read_csv(csv_path)

# Convert columns to datetime
df["date"] = pd.to_datetime(df["date"])
df["detected_peak_date"] = pd.to_datetime(df["detected_peak_date"])

# Sort by date
df = df.sort_values("date")

# Get peak date
peak_date = df["detected_peak_date"].iloc[0]

# Plot campaign_index over time
plt.figure(figsize=(12, 6))
plt.plot(df["date"], df["campaign_index"], marker="o", label="Campaign index")
plt.axvline(peak_date, linestyle="--", label="Detected peak")

# Add labels and title
plt.xlabel("Date")
plt.ylabel("Campaign index")
plt.title("Dry January - campaign_index over time")

# Show legend
plt.legend()

# Rotate x-axis labels for readability
plt.xticks(rotation=45)

# Improve layout
plt.tight_layout()

# Show plot
plt.show()