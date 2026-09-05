"""
Week 4 Task: Data Visualization and Storytelling
Data Science with Python — Customer Churn Dataset

Creates three distinct visualizations (line plot, bar chart, heat map)
with interpretations, building on the cleaned dataset and modeling
results from Weeks 2 and 3.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

df = pd.read_csv("customer_churn_cleaned.csv")
df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)

# -----------------------------
# Visualization 1: Line Plot
# Churn rate trend across tenure bands
# -----------------------------
bins = [0, 6, 12, 24, 36, 48, 60, 72]
labels = ["0-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
df["TenureBand"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)

churn_by_tenure_band = df.groupby("TenureBand", observed=True)["ChurnFlag"].mean() * 100

plt.figure(figsize=(7.5, 4.3))
plt.plot(churn_by_tenure_band.index.astype(str), churn_by_tenure_band.values,
         marker="o", linewidth=2.5, color="#C62828")
plt.fill_between(range(len(churn_by_tenure_band)), churn_by_tenure_band.values, alpha=0.15, color="#C62828")
plt.title("Churn Rate Trend Across Customer Tenure Bands")
plt.xlabel("Tenure Band (months)")
plt.ylabel("Churn Rate (%)")
for i, v in enumerate(churn_by_tenure_band.values):
    plt.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("viz1_line_churn_by_tenure_band.png", dpi=150)
plt.close()
print("Churn rate by tenure band:\n", churn_by_tenure_band)

# -----------------------------
# Visualization 2: Bar Chart
# Average monthly charges by contract type, split by churn status
# -----------------------------
avg_charges = df.groupby(["Contract", "Churn"], observed=True)["MonthlyCharges"].mean().unstack()

plt.figure(figsize=(7.5, 4.3))
avg_charges.plot(kind="bar", ax=plt.gca(), color=["#2E7D32", "#C62828"])
plt.title("Average Monthly Charges by Contract Type and Churn Status")
plt.xlabel("Contract Type")
plt.ylabel("Average Monthly Charges ($)")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()
plt.savefig("viz2_bar_charges_by_contract_churn.png", dpi=150)
plt.close()
print("\nAverage charges by contract and churn:\n", avg_charges)

# -----------------------------
# Visualization 3: Heat Map
# Churn rate across Contract x InternetService (cross-tab)
# -----------------------------
pivot = df.pivot_table(values="ChurnFlag", index="Contract", columns="InternetService",
                        aggfunc="mean", observed=True) * 100

plt.figure(figsize=(7, 4.5))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds", cbar_kws={"label": "Churn Rate (%)"})
plt.title("Churn Rate (%) by Contract Type and Internet Service")
plt.tight_layout()
plt.savefig("viz3_heatmap_contract_internet.png", dpi=150)
plt.close()
print("\nChurn rate heatmap (Contract x InternetService):\n", pivot)

print("\nAll three visualizations saved successfully.")
