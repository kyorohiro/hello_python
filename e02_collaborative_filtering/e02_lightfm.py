import numpy as np
from scipy.sparse import coo_matrix
from lightfm import LightFM

# ===== 疑似データ =====
# ユーザー 5人（id: 0〜4）
# アイテム 10個（id: 0〜9）
# 「ユーザー u がアイテム i を気に入った（閲覧/購入した）」というログをいくつか作る

interactions_data = np.array([
    1, 1, 1, 1, 1, 1
])
user_ids = np.array([
    0, 0, 1, 1, 2, 3
])
item_ids = np.array([
    0, 1, 1, 2, 3, 4
])

num_users = 5
num_items = 10

# COO 形式の疎行列にする
interactions = coo_matrix(
    (interactions_data, (user_ids, item_ids)),
    shape=(num_users, num_items)
)

print("interactions shape:", interactions.shape)

# ===== モデル作成 & 学習 =====
model = LightFM(loss="warp")  # implicit feedback なら WARP が鉄板
model.fit(interactions, epochs=20, num_threads=2)

# ===== user 0 に対するレコメンド =====
user_id = 0
scores = model.predict(user_id, np.arange(num_items))
ranked_items = np.argsort(-scores)

print(f"user {user_id} scores:", scores)
print(f"user {user_id} recommended items (best->worst):", ranked_items)
