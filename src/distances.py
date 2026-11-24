"""
Stage 4: Distance Computation
Computes pairwise cosine distances for each precision format.
Records intra-speaker and inter-speaker distances and timing.
"""
import numpy as np, pandas as pd, time, yaml, os
from itertools import combinations
from scipy.spatial.distance import cosine

params = yaml.safe_load(open("params.yaml"))
formats = params["precision"]["formats"]

meta = pd.read_csv("data/embeddings/metadata.csv")
os.makedirs("data/distances", exist_ok=True)

all_results = []

for fmt in formats:
    reps = np.load(f"data/precision/embeddings_{fmt}.npy").astype(np.float64)
    t0 = time.time()
    for i, j in combinations(range(len(meta)), 2):
        dist = cosine(reps[i], reps[j])
        pair_type = "intra-speaker" if meta.iloc[i]["speaker"] == meta.iloc[j]["speaker"] else "inter-speaker"
        all_results.append({"precision": fmt, "type": pair_type, "distance": dist})
    elapsed = round(time.time() - t0, 4)
    print(f"{fmt:8s} done in {elapsed}s")

df = pd.DataFrame(all_results)
df.to_csv("data/distances/distances.csv", index=False)

summary = df.groupby(["precision","type"])["distance"].mean().unstack()
summary["ratio"] = summary["inter-speaker"] / summary["intra-speaker"]
summary.to_csv("data/distances/summary.csv")
print(summary)
