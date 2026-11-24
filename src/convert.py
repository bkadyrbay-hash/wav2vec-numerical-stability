"""
Stage 3: Precision Conversion
Converts float32 embeddings to float64, float16, and int8.
int8 uses linear (min-max) quantisation: maps the global range to [-128, 127],
then dequantises back to float for distance computation.
"""
import numpy as np, os, yaml

params = yaml.safe_load(open("params.yaml"))
formats = params["precision"]["formats"]

base = np.load("data/embeddings/embeddings_float32.npy")
os.makedirs("data/precision", exist_ok=True)

def quantise_int8(x):
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    if mx == mn:
        return np.zeros_like(x, dtype=np.float32)
    scale = (mx - mn) / 255
    q = np.round((x - mn) / scale - 128).astype(np.int8)
    # dequantise for distance computation
    return (q.astype(np.float32) + 128) * scale + mn

converters = {
    "float64": lambda x: x.astype(np.float64),
    "float32": lambda x: x.astype(np.float32),
    "float16": lambda x: x.astype(np.float16),
    "int8":    quantise_int8,
}

for fmt in formats:
    converted = converters[fmt](base)
    np.save(f"data/precision/embeddings_{fmt}.npy", converted)
    mb = converted.nbytes / 1e6
    print(f"{fmt:8s}: shape={converted.shape}  dtype={converted.dtype}  size={mb:.4f} MB")
