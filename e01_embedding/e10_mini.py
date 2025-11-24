# =========================================
# ライブラリ
# =========================================
# !pip install -q sentence-transformers

from sentence_transformers import SentenceTransformer
import numpy as np


# =========================================
# 商品データ（完全版）
# ※ 「鍋＋日常食材」ベースの簡易サンプル
# =========================================
products = [
    # --- 野菜 ---
    {"id": 1, "name": "白菜 1/4カット", "category": "野菜", "tags": ["鍋", "冬野菜"], 
     "description": "鍋料理に欠かせない甘みのある白菜。"},
    {"id": 2, "name": "長ねぎ 2本", "category": "野菜", "tags": ["鍋", "薬味"],
     "description": "鍋に入れると風味が増す長ねぎ。"},
    {"id": 3, "name": "しめじ 1パック", "category": "きのこ", "tags": ["鍋"],
     "description": "鍋の具材として使いやすいしめじ。"},
    {"id": 4, "name": "えのき茸 100g", "category": "きのこ", "tags": ["鍋"],
     "description": "鍋の定番食材、えのき茸。"},

    # --- 肉 ---
    {"id": 10, "name": "豚バラ肉 200g", "category": "肉", "tags": ["鍋", "豚"], 
     "description": "寄せ鍋やキムチ鍋に合う豚バラ肉。"},
    {"id": 11, "name": "鶏もも肉 300g", "category": "肉", "tags": ["鍋", "水炊き"], 
     "description": "水炊きに最適な鶏もも肉。"},

    # --- 豆腐 ---
    {"id": 20, "name": "木綿豆腐 1丁", "category": "豆腐", "tags": ["鍋"],
     "description": "鍋にしっかり崩れず入れられる木綿豆腐。"},
    {"id": 21, "name": "絹豆腐 1丁", "category": "豆腐", "tags": ["鍋"],
     "description": "柔らかい食感の絹豆腐。"},

    # --- その他 ---
    {"id": 22, "name": "しらたき 200g", "category": "麺類", "tags": ["鍋"],
     "description": "鍋のかさ増しにちょうどいいしらたき。"},

    # --- スープ類 ---
    {"id": 30, "name": "寄せ鍋スープ 醤油味", "category": "スープ", "tags": ["鍋"],
     "description": "寄せ鍋用の醤油ベーススープ。"},
    {"id": 31, "name": "キムチ鍋の素", "category": "スープ", "tags": ["鍋", "キムチ"],
     "description": "ピリ辛のキムチ味鍋の素。"},
    {"id": 32, "name": "豆乳鍋スープ", "category": "スープ", "tags": ["鍋", "豆乳"],
     "description": "まろやかな豆乳味の鍋スープ。"},

    # --- しめ ---
    {"id": 40, "name": "鍋用 中華麺", "category": "麺類", "tags": ["鍋", "しめ"],
     "description": "鍋のしめに使える中華麺。"},
    {"id": 41, "name": "鍋の〆 雑炊セット", "category": "米", "tags": ["鍋", "しめ"],
     "description": "出汁が効いた雑炊が作れるセット。"},
]


# =========================================
# embedding モデル
# =========================================
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def product_to_text(p):
    tags_text = " ".join(p.get("tags", []))
    return f"{p['name']}。カテゴリ: {p['category']}。タグ: {tags_text}。説明: {p['description']}"

# 商品テキスト → 埋め込み
product_texts = [product_to_text(p) for p in products]
product_embeddings = model.encode(product_texts, normalize_embeddings=True)


# =========================================
# 「この商品も買いませんか？」（商品→商品）
# =========================================
def recommend_related_products(product_name, top_k=5):
    """
    商品名に意味的に近い商品を推薦する（＝この商品も買いませんか？）
    """

    # 入力商品の embedding
    query_emb = model.encode([product_name], normalize_embeddings=True)[0]

    # 類似度
    scores = product_embeddings @ query_emb

    # 自分自身の商品 index を特定
    self_idx = None
    for i, p in enumerate(products):
        if p["name"] == product_name:
            self_idx = i
            break

    # スコア順に top_k 取得
    idxs = np.argsort(scores)[::-1]

    results = []
    for idx in idxs:
        if idx == self_idx:
            continue  # 自分を除外
        p = products[idx]
        results.append({
            "id": p["id"],
            "name": p["name"],
            "score": float(scores[idx])
        })
        if len(results) >= top_k:
            break

    return results

# =========================================
# 複数の商品名をまとめて関連商品推薦
# =========================================
def recommend_related_products_multi(product_names, top_k=5):
    """
    product_names: ["豚バラ肉 200g", "しらたき 200g"] のようなリスト
    """
    # --- 入力商品それぞれを embedding ---
    query_embs = []
    for name in product_names:
        emb = model.encode([name], normalize_embeddings=True)[0]
        query_embs.append(emb)

    # --- 平均を取る（複数商品の意味の“中心”を取る）---
    query_emb = np.mean(query_embs, axis=0)

    # --- 類似度 ---
    scores = product_embeddings @ query_emb

    # --- 自分自身(購入済み商品)を除外 ---
    exclude_idxs = set()
    for name in product_names:
        for i, p in enumerate(products):
            if p["name"] == name:
                exclude_idxs.add(i)

    # --- スコア順に top_k ---
    idxs = np.argsort(scores)[::-1]

    results = []
    for idx in idxs:
        if idx in exclude_idxs:
            continue  # 自分自身を除外
        p = products[idx]
        results.append({
            "id": p["id"],
            "name": p["name"],
            "score": float(scores[idx])
        })
        if len(results) >= top_k:
            break

    return results

# =========================================
# ★ 動作確認 ★
# =========================================
print("=== 豚バラ肉 200g を買うなら、この商品もどうですか？ ===")
for r in recommend_related_products("豚バラ肉 200g", top_k=5):
    print(f"{r['score']:.3f}  {r['name']} (id={r['id']})")


print("=== 豚バラ肉 + しらたき を買うなら、この商品もどうですか？ ===")
for r in recommend_related_products_multi(["豚バラ肉 200g", "しらたき 200g"], top_k=5):
    print(f"{r['score']:.3f}  {r['name']} (id={r['id']})")
