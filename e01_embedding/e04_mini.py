from sentence_transformers import SentenceTransformer
import hnswlib
import numpy as np

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

items = [
    "低脂肪牛乳 1L",
    "北海道産バター 200g",
    "無糖ヨーグルト 450g",
    "醤油 1L",
    "豚ロース 300g",
]

emb = model.encode(items)
print("shape =", emb.shape)

###
###
##
# embedding の準備
vectors = emb.astype(np.float32)
dim = vectors.shape[1]

# HNSW index 作成
p = hnswlib.Index(space='cosine', dim=dim)
p.init_index(max_elements=len(vectors), ef_construction=200, M=16)
p.add_items(vectors)

# 検索
q = model.encode(["無糖ヨーグルト"]).astype(np.float32)
labels, distances = p.knn_query(q, k=3)
print(labels, distances)
