"""
Stage 1: Data Preparation
Scans corpus directory, builds a CSV index of all .wav files with speaker labels.
"""
import glob, os, yaml, pandas as pd
from pathlib import Path

params = yaml.safe_load(open("params.yaml"))
corpus_dir = params["data"]["corpus_dir"]
sample_size = params["data"]["sample_size"]
random_state = params["data"]["random_state"]

wav_files = glob.glob(f"{corpus_dir}/**/*.wav", recursive=True)
rows = []
for path in wav_files:
    name = Path(path).stem
    speaker = name.split("_")[0]
    rows.append({"path": path, "speaker": speaker, "filename": name})

df = pd.DataFrame(rows)
sample = df.sample(min(sample_size, len(df)), random_state=random_state).reset_index(drop=True)

os.makedirs("data/processed", exist_ok=True)
sample.to_csv("data/processed/file_index.csv", index=False)
print(f"Saved {len(sample)} files to data/processed/file_index.csv")
