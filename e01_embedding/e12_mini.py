# =====================================
# インストール（ColabならそのままOK）
# =====================================
# !pip install -q sentence-transformers

from sentence_transformers import SentenceTransformer
import numpy as np

# =====================================
# ① レシピデータ（レシピサイトから取ってきた想定）
#    {name: 料理名, elems: [材料]
# =====================================
recipes = [
    {
        "name": "寄せ鍋",
        "elems": ["白菜", "長ねぎ", "豆腐", "豚肉", "しめじ", "しらたき"]
    },
    {
        "name": "キムチ鍋",
        "elems": ["白菜", "豚肉", "豆腐", "キムチの素", "長ねぎ"]
    },
    {
        "name": "水炊き",
        "elems": ["鶏肉", "白菜", "豆腐", "長ねぎ", "昆布"]
    },
    {
        "name": "味噌汁",
        "elems": ["味噌", "豆腐", "わかめ", "だし", "長ねぎ"]
    },
    {
        "name": "豚汁",
        "elems": ["豚肉", "大根", "にんじん", "味噌", "ごぼう", "長ねぎ"]
    },
]

# =====================================
# ② SentenceTransformer モデル読み込み
# =====================================
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# =====================================
# ③ {name, elems[]} → 1本の文章にする関数
# =====================================
def recipe_to_text(recipe):
    """
    {name: 料理名, elems: [材料]} を
    "料理名。材料: ..." という文章にする
    """
    ing = "、".join(recipe["elems"])
    return f"{recipe['name']}。材料: {ing}"

# レシピごとにテキスト化
recipe_texts = [recipe_to_text(r) for r in recipes]

# embedding 生成（ここでベクトル空間に入る）
recipe_embeddings = model.encode(recipe_texts, normalize_embeddings=True)

# =====================================
# ④ 手持ちの材料 → 作れそうな料理をレコメンド
# =====================================
def recommend_recipes_from_ingredients(ingredients_list, top_k=5):
    """
    ingredients_list: ["白菜", "豆腐", "鶏肉"] みたいなリスト
    を渡すと、作れそうな料理をスコア順に返す
    """
    # 材料リスト → 1本のテキスト
    query_text = "、".join(ingredients_list)
    # テキスト → embedding
    query_emb = model.encode([query_text], normalize_embeddings=True)[0]

    # コサイン類似度相当（正規化してるので内積でOK）
    scores = recipe_embeddings @ query_emb

    # スコア順に top_k 件
    idxs = np.argsort(scores)[::-1][:top_k]

    results = []
    for i in idxs:
        results.append(
            {
                "name": recipes[i]["name"],
                "score": float(scores[i]),
                "ingredients": recipes[i]["elems"],
            }
        )
    return results

# =====================================
# ⑤ 「これも買えば？」候補の材料を出す
#    上位レシピたちから「足りていない材料」を集計して提案
# =====================================
def recommend_extra_ingredients(ingredients_list, top_k_recipes=3, top_k_ingredients=5):
    """
    ingredients_list: 手持ちの材料リスト
    top_k_recipes: 参考にする上位レシピ数
    top_k_ingredients: 提案する材料数

    戻り値: [{name: 材料名, score: 重み}, ...]
    """
    # まず「作れそうな料理」を取得
    recs = recommend_recipes_from_ingredients(ingredients_list, top_k=top_k_recipes)

    have = set(ingredients_list)
    missing_scores = {}  # 材料名 → スコア（どのレシピでどれだけ重要そうか）

    for r in recs:
        sim = r["score"]  # 料理との類似度を重みとして使う
        for ing in r["ingredients"]:
            if ing in have:
                continue  # すでに持っている材料はスキップ
            # 足りない材料にスコア加算
            missing_scores[ing] = missing_scores.get(ing, 0.0) + sim

    # スコア順に材料を並べる
    sorted_ings = sorted(missing_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for name, score in sorted_ings[:top_k_ingredients]:
        results.append({"name": name, "score": float(score)})
    return results

# =====================================
# ⑥ 動作確認
# =====================================
my_ingredients = ["白菜", "豆腐", "鶏肉"]  # 手持ちの材料

print("=== 手持ちの材料:", my_ingredients, "→ 作れそうな料理 ===")
for r in recommend_recipes_from_ingredients(my_ingredients, top_k=3):
    print(f"{r['score']:.3f}  {r['name']}  材料: {r['ingredients']}")

print("\n=== これも買えば？（足りてなさそうな材料） ===")
for e in recommend_extra_ingredients(my_ingredients, top_k_recipes=3, top_k_ingredients=5):
    print(f"{e['score']:.3f}  {e['name']}")
