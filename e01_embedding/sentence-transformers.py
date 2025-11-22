from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm

def cosine_sim(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b)))

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

words = ["king", "queen", "man", "woman", "prince", "princess", "apple", "car"]
embs = model.encode(words)

# index を分かりやすく
idx = {w: i for i, w in enumerate(words)}

king  = embs[idx["king"]]
queen = embs[idx["queen"]]
man   = embs[idx["man"]]
woman = embs[idx["woman"]]

vec = king - man + woman

print("=== cos類似度（king-man+woman vs 各単語）===")
for w in words:
    sim = cosine_sim(vec, embs[idx[w]])
    print(f"{w:8s} : {sim:.3f}")
