"""
Stage 5: Visualisation
Produces boxplots and bar chart of intra/inter distances across precision levels.
"""
import pandas as pd, matplotlib.pyplot as plt, os

df = pd.read_csv("data/distances/distances.csv")
summary = pd.read_csv("data/distances/summary.csv", index_col=0)
os.makedirs("data/figures", exist_ok=True)

formats = df["precision"].unique()

# Boxplots per precision
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, fmt in zip(axes.flatten(), formats):
    subset = df[df["precision"] == fmt]
    subset.boxplot(column="distance", by="type", ax=ax)
    ax.set_title(f"Cosine distances: {fmt}")
    ax.set_xlabel("Pair type")
    ax.set_ylabel("Cosine distance")
plt.suptitle("")
plt.tight_layout()
plt.savefig("data/figures/boxplots.png", dpi=150)
plt.close()
print("Saved: data/figures/boxplots.png")

# Bar chart: intra vs inter by format
summary[["intra-speaker","inter-speaker"]].plot(kind="bar", figsize=(8,5))
plt.title("Average cosine distance by precision")
plt.xlabel("Precision")
plt.ylabel("Cosine distance")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("data/figures/barchart.png", dpi=150)
plt.close()
print("Saved: data/figures/barchart.png")

# Ratio plot
plt.figure(figsize=(7,4))
plt.plot(summary.index, summary["ratio"], marker="o", linewidth=2, color="purple")
plt.axhline(summary["ratio"].iloc[0], linestyle="--", color="gray", label="float64 reference")
plt.ylabel("Inter / Intra ratio")
plt.title("Speaker separability across precision levels")
plt.legend()
plt.tight_layout()
plt.savefig("data/figures/ratio.png", dpi=150)
plt.close()
print("Saved: data/figures/ratio.png")
