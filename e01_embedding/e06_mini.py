# ===================================
# 必要ライブラリ
# ===================================
# !pip install -q sentence-transformers

from sentence_transformers import SentenceTransformer
import numpy as np


# ===================================
# 商品データ（鍋特化の簡易データ）
# ===================================
products = [
    # ---- 野菜 ----
    {"id": 1, "name": "白菜 1/4カット", "category": "野菜", "tags": ["鍋", "冬野菜"], 
     "description": "鍋料理に欠かせない甘みのある白菜。"},
    {"id": 2, "name": "長ねぎ 2本", "category": "野菜", "tags": ["鍋", "薬味"], 
     "description": "鍋に入れると風味が増す長ねぎ。"},
    {"id": 3, "name": "しめじ 1パック", "category": "きのこ", "tags": ["鍋"], 
     "description": "鍋の具材として使いやすいしめじ。"},
    {"id": 4, "name": "えのき茸 100g", "category": "きのこ", "tags": ["鍋"], 
     "description": "鍋の定番食材、えのき茸。"},

    # ---- 肉 ----
    {"id": 10, "name": "豚バラ肉 200g", "category": "肉", "tags": ["鍋", "豚"], 
     "description": "寄せ鍋やキムチ鍋に合う豚バラ肉。"},
    {"id": 11, "name": "鶏もも肉 300g", "category": "肉", "tags": ["鍋", "水炊き"], 
     "description": "水炊きに最適な鶏もも肉。"},

    # ---- 豆腐 / その他 ----
    {"id": 20, "name": "木綿豆腐 1丁", "category": "豆腐", "tags": ["鍋"], 
     "description": "鍋にしっかり崩れず入れられる木綿豆腐。"},
    {"id": 21, "name": "絹豆腐 1丁", "category": "豆腐", "tags": ["鍋"], 
     "description": "柔らかい食感の絹豆腐。"},
    {"id": 22, "name": "しらたき 200g", "category": "麺類", "tags": ["鍋"], 
     "description": "鍋のかさ増しにちょうどいいしらたき。"},

    # ---- スープ / 調味料 ----
    {"id": 30, "name": "寄せ鍋スープ 醤油味", "category": "スープ", "tags": ["鍋"], 
     "description": "寄せ鍋用の醤油ベーススープ。"},
    {"id": 31, "name": "キムチ鍋の素", "category": "スープ", "tags": ["鍋", "キムチ"], 
     "description": "ピリ辛のキムチ味鍋の素。"},
    {"id": 32, "name": "豆乳鍋スープ", "category": "スープ", "tags": ["鍋", "豆乳"], 
     "description": "まろやかな豆乳味の鍋スープ。"},

    # ---- しめ ----
    {"id": 40, "name": "鍋用 中華麺", "category": "麺類", "tags": ["鍋", "しめ"], 
     "description": "鍋のしめに使える中華麺。"},
    {"id": 41, "name": "鍋の〆 雑炊セット", "category": "米", "tags": ["鍋", "しめ"], 
     "description": "出汁が効いた雑炊が作れるセット。"},
]


# ===================================
# 埋め込み生成
# ===================================
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def product_to_text(p):
    tags_text = " ".join(p.get("tags", []))
    return f"{p['name']}。カテゴリ:{p['category']}。タグ:{tags_text}。説明:{p['description']}"

product_texts = [product_to_text(p) for p in products]
product_embeddings = model.encode(product_texts, normalize_embeddings=True)


# ===================================
# 鍋関連フィルター
# ===================================
def is_hotpot_related(p):
    hotpot_keywords = [
        "鍋", "スープ", "白菜", "ねぎ", "ネギ", "長ねぎ",
        "しめじ", "えのき", "豆腐", "しらたき",
        "豚", "鶏", "肉", "〆", "雑炊", "中華麺"
    ]
    text = p["name"] + " " + p["description"] + " " + " ".join(p.get("tags", []))
    return any(k in text for k in hotpot_keywords)


# ===================================
# 抜けている鍋具材を提案
# ===================================
def suggest_missing_hotpot_items(
    dish_text: str,
    cart_product_ids,
    products,
    product_embeddings,
    top_k: int = 5,
):
    cart_product_ids = set(cart_product_ids)

    # 鍋の説明を embedding
    query_emb = model.encode([dish_text], normalize_embeddings=True)[0]

    # 類似度計算
    scores = product_embeddings @ query_emb

    candidates = []
    for idx, p in enumerate(products):
        if p["id"] in cart_product_ids:
            continue
        if not is_hotpot_related(p):
            continue
        candidates.append((scores[idx], p))

    candidates.sort(key=lambda x: x[0], reverse=True)

    return [
        {"id": p["id"], "name": p["name"], "score": float(score)}
        for score, p in candidates[:top_k]
    ]


# ===================================
# 動作例
# ===================================
dish = "今日は家族で寄せ鍋を作りたい。野菜多めでヘルシーにしたい。"
#dish = "寄せ鍋"
cart_ids = [30, 1]  # すでに「寄せ鍋スープ」と「白菜」を購入済み

results = suggest_missing_hotpot_items(
    dish_text=dish,
    cart_product_ids=cart_ids,
    products=products,
    product_embeddings=product_embeddings,
    top_k=5,
)

for r in results:
    print(f"{r['score']:.3f}  {r['name']} (id={r['id']})")
