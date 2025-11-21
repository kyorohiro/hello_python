# 
# !pip install -U sentence-transformers
# 
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
import numpy as np
model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = [
    "これはペンです。",
    "これは鉛筆です。",
    "私はカレーが好きです。",
    "サッカーはスポーツです。"
]

embs = model.encode(sentences)

print("埋め込みの形状:", embs.shape)  # (文の数, 次元数)
print(embs);
# 1つ目の文の最初の10要素だけ表示
print("1つ目の文のembedding（先頭10次元）:")
print(embs[0][:10])


#
# コサイン類似度の計算と表示 
#

def cosine_sim(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b)))

for i, s1 in enumerate(sentences):
    for j, s2 in enumerate(sentences):
        if i >= j:
            continue
        sim = cosine_sim(embs[i], embs[j])
        print(f"「{s1}」 vs 「{s2}」 => cos類似度: {sim:.3f}")
