import numpy as np
from scipy.sparse import coo_matrix
from lightfm import LightFM

# ===== 疑似データ =====
interactions_data = np.array([1, 1, 1, 1, 1, 1])
user_ids = np.array([0, 0, 1, 1, 2, 3])
item_ids = np.array([0, 1, 1, 2, 3, 4])

num_users = 5
num_items = 10

interactions = coo_matrix(
    (interactions_data, (user_ids, item_ids)),
    shape=(num_users, num_items)
)

print("interactions shape:", interactions.shape)

# ===== モデル作成 & 学習 =====
model = LightFM(loss="warp")  # implicit 用
model.fit(interactions, epochs=20, num_threads=2)


# LightFM から埋め込みを取り出す
user_biases, user_embeddings = model.get_user_representations()
item_biases, item_embeddings = model.get_item_representations()

print("user_embeddings shape:", user_embeddings.shape)  # (num_users, k)
print("item_embeddings shape:", item_embeddings.shape)  # (num_items, k)

# ===== HNSW インデックス作成 =====
import hnswlib

dim = item_embeddings.shape[1]      # ベクトル次元数（LightFM の latent_dim）
num_elements = item_embeddings.shape[0]  # アイテム数（num_items）

# ===== HNSW インデックスを作成 =====
p = hnswlib.Index(space='cosine', dim=dim)  # ここでは cosine 類似度ベース

p.init_index(
    max_elements=num_elements,
    ef_construction=200,
    M=16,
)

# アイテムベクトルを ANN に登録
item_ids = np.arange(num_elements)  # 0,1,2,... の item index
p.add_items(item_embeddings, item_ids)

# 検索時の精度/速度トレードオフ
p.set_ef(50)  # 大きくすると精度↑ 速度↓




# おすすめを出す

# 例: user_id = 0 の埋め込みベクトルを取得
user_id = 0
user_vec = user_embeddings[user_id].reshape(1, -1)  # 形を (1, dim) に

# ANNで「このユーザーに近いアイテム Top-K」を取得
K = 5
labels, distances = p.knn_query(user_vec, k=K)

# labels[0] に「近い item index」が並ぶ
print("user", user_id, "近い item indices:", labels[0])
print("distances (cosine距離):", distances[0])

# LightFM の predict と比べてみる（検証用）
from numpy.linalg import norm

scores_lightfm = model.predict(user_id, np.arange(num_items))
ranked_by_lightfm = np.argsort(-scores_lightfm)

print("LightFM ranking:", ranked_by_lightfm)
print("ANN topK:", labels[0])
