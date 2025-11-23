from sentence_transformers import SentenceTransformer

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
