# ① 必要なライブラリのインストール（Colab なら最初の1回だけ）
# !pip install -q sentence-transformers

# ② サンプル商品データ（実際はここをあなたの配列に差し替え）
products = [
    {
        "id": 123,
        "name": "低脂肪ヨーグルト いちご味",
        "category": "ヨーグルト",
        "tags": ["低脂肪", "朝食", "ダイエット"],
        "description": "朝に食べやすい低脂肪タイプのストロベリーヨーグルトです。"
    },
    {
        "id": 124,
        "name": "プレーンヨーグルト",
        "category": "ヨーグルト",
        "tags": ["朝食"],
        "description": "シンプルな無糖タイプのプレーンヨーグルトです。"
    },
    {
        "id": 200,
        "name": "いちごジャム",
        "category": "ジャム",
        "tags": ["パン", "スイーツ"],
        "description": "パンやヨーグルトのトッピングに合う甘いいちごジャムです。"
    },
]

# ③ モデル読み込み
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ④ 商品1件を「文章」に変換する関数
#   name を中心に、category や tags, description も足して意味をリッチにしています
def product_to_text(p):
    tags_text = "、".join(p.get("tags", []))
    return (
        f"{p['name']}。"
        f"カテゴリ: {p.get('category', '')}。"
        f"タグ: {tags_text}。"
        f"説明: {p.get('description', '')}"
    )

# ⑤ 全商品の埋め込みをあらかじめ計算
product_texts = [product_to_text(p) for p in products]
product_embeddings = model.encode(product_texts, normalize_embeddings=True)  # cos類似度用に正規化

# ⑥ 今ある材料(テキスト)からおすすめ商品を出す関数
from typing import List, Dict

def recommend_products_from_ingredients(
    ingredients_text: str,
    products: List[Dict],
    product_embeddings: np.ndarray,
    top_k: int = 5,
):
    """
    ingredients_text: 「今ある材料」を書いた日本語の文章
        例: "いちごとヨーグルトとはちみつがあります"
    """
    # 材料の文章を埋め込み
    query_emb = model.encode([ingredients_text], normalize_embeddings=True)[0]

    # コサイン類似度 = 正規化したベクトル同士の内積
    scores = product_embeddings @ query_emb  # shape: (n_products,)

    # 類似度の高い順に top_k 件を取得
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idx:
        p = products[idx]
        results.append({
            "id": p["id"],
            "name": p["name"],
            "score": float(scores[idx]),
        })
    return results

# ⑦ 使ってみる
# 例: 材料として「いちご」と「ヨーグルト」がある
ingredients = "冷蔵庫に低脂肪のヨーグルトといちごがある"
recommendations = recommend_products_from_ingredients(
    ingredients_text=ingredients,
    products=products,
    product_embeddings=product_embeddings,
    top_k=3,
)

for r in recommendations:
    print(f"{r['score']:.3f}  {r['name']} (id={r['id']})")
