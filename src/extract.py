"""
Stage 2: Feature Extraction
Extracts wav2vec2 embeddings for each recording via mean pooling.
Saves as float32 numpy array.
"""
import yaml, numpy as np, pandas as pd, librosa, torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from pathlib import Path
import os

params = yaml.safe_load(open("params.yaml"))
model_name = params["model"]["name"]
sr = params["model"]["sample_rate"]

df = pd.read_csv("data/processed/file_index.csv")

processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name).eval()

def get_embedding(path):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state.mean(dim=1).squeeze().numpy().astype(np.float32)

os.makedirs("data/embeddings", exist_ok=True)
embeddings = []
for _, row in df.iterrows():
    emb = get_embedding(row["path"])
    embeddings.append(emb)
    print(f"  extracted: {row['filename']}")

arr = np.stack(embeddings)  # (N, 768) float32
np.save("data/embeddings/embeddings_float32.npy", arr)
df.to_csv("data/embeddings/metadata.csv", index=False)
print(f"Saved embeddings shape: {arr.shape}")
