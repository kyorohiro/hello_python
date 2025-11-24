# =====================================
# インストール（ColabならそのままOK）
# =====================================
# !pip install -q sentence-transformers

from sentence_transformers import SentenceTransformer
import numpy as np

# =====================================
# ① レシピデータ
#    {name: 料理名, genre: ジャンル, elems: [材料]}
# =====================================
recipes = [
    {
        "name": "寄せ鍋",
        "genre": "和風",
        "elems": ["白菜", "長ねぎ", "豆腐", "豚肉", "しめじ", "しらたき"]
    },
    {
        "name": "キムチ鍋",
        "genre": "中華",
        "elems": ["白菜", "豚肉", "豆腐", "キムチの素", "長ねぎ"]
    },
    {
        "name": "水炊き",
        "genre": "和風",
        "elems": ["鶏肉", "白菜", "豆腐", "長ねぎ", "昆布"]
    },
    {
        "name": "味噌汁",
        "genre": "和風",
        "elems": ["味噌", "豆腐", "わかめ", "だし", "長ねぎ"]
    },
    {
        "name": "豚汁",
        "genre": "和風",
        "elems": ["豚肉", "大根", "にんじん", "味噌", "ごぼう", "長ねぎ"]
    },
    {
        "name": "ペペロンチーノ",
        "genre": "洋風",
        "elems": ["スパゲッティ", "にんにく", "オリーブオイル", "唐辛子", "塩"]
    },
    {
        "name": "トマトパスタ",
        "genre": "洋風",
        "elems": ["スパゲッティ", "トマトソース", "にんにく", "オリーブオイル"]
    },
    {
        "name": "麻婆豆腐",
        "genre": "中華",
        "elems": ["豆腐", "豚ひき肉", "ねぎ", "豆板醤", "甜麺醤"]
    },
]

# =====================================
# ② SentenceTransformer モデル読み込み
# =====================================
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# =====================================
# ③ {name, genre, elems[]} → 1本の文章にする関数
# =====================================
def recipe_to_text(recipe):
    """
    {name: 料理名, genre: ジャンル, elems: [材料]} を
    "料理名。ジャンル: 和風。材料: ..." という文章にする
    """
    ing = "、".join(recipe["elems"])
    return f"{recipe['name']}。ジャンル: {recipe['genre']}。材料: {ing}"

# レシピごとにテキスト化
recipe_texts = [recipe_to_text(r) for r in recipes]

# embedding 生成（ここでベクトル空間に入る）
recipe_embeddings = model.encode(recipe_texts, normalize_embeddings=True)

# =====================================
# ④ 手持ちの材料 → 作れそうな料理をレコメンド
#    preferred_genres でジャンル指定が可能
#    例: ["和風"] / ["和風", "中華"] / None(全部)
# =====================================
def recommend_recipes_from_ingredients(ingredients_list, preferred_genres=None,
                                       top_k=5):
    """
    ingredients_list: ["白菜", "豆腐", "鶏肉"] みたいなリスト
    preferred_genres: ["和風"], ["和風","中華"], None など
    """
    # 材料リスト → 1本のテキスト
    query_text = "、".join(ingredients_list)
    # テキスト → embedding
    query_emb = model.encode([query_text], normalize_embeddings=True)[0]

    # コサイン類似度相当（正規化してるので内積でOK）
    scores = recipe_embeddings @ query_emb

    # スコア順にソート
    idxs_all = np.argsort(scores)[::-1]

    results = []
    for i in idxs_all:
        r = recipes[i]
        # ジャンル指定があればフィルタ
        if preferred_genres is not None and r["genre"] not in preferred_genres:
            continue
        results.append(
            {
                "name": r["name"],
                "genre": r["genre"],
                "score": float(scores[i]),
                "ingredients": r["elems"],
            }
        )
        if len(results) >= top_k:
            break
    return results

# =====================================
# ⑤ 「これも買えば？」候補の材料を出す
#    上位レシピたちから「足りていない材料」を集計して提案
# =====================================
def recommend_extra_ingredients(ingredients_list,
                                preferred_genres=None,
                                top_k_recipes=3,
                                top_k_ingredients=5):
    """
    ingredients_list: 手持ちの材料リスト
    preferred_genres: ["和風"], ["洋風"], None など
    """
    # まず「作れそうな料理」をジャンル込みで取得
    recs = recommend_recipes_from_ingredients(
        ingredients_list,
        preferred_genres=preferred_genres,
        top_k=top_k_recipes,
    )

    have = set(ingredients_list)
    missing_scores = {}  # 材料名 → スコア（どのレシピでどれだけ重要そうか）

    for r in recs:
        sim = r["score"]  # 料理との類似度を重みとして使う
        for ing in r["ingredients"]:
            if ing in have:
                continue  # すでに持っている材料はスキップ
            missing_scores[ing] = missing_scores.get(ing, 0.0) + sim

    # スコア順に材料を並べる
    sorted_ings = sorted(missing_scores.items(), key=lambda x: x[1],
                         reverse=True)

    results = []
    for name, score in sorted_ings[:top_k_ingredients]:
        results.append({"name": name, "score": float(score)})
    return results

# =====================================
# ⑥ 動作確認
# =====================================
# 手持ちの材料
my_ingredients = ["白菜", "豆腐", "鶏肉"]

print("=== 手持ちの材料:", my_ingredients, "→ 作れそうな料理（ジャンル指定なし） ===")
for r in recommend_recipes_from_ingredients(my_ingredients, preferred_genres=None,
                                            top_k=5):
    print(f"{r['score']:.3f}  [{r['genre']}] {r['name']}  材料: {r['ingredients']}")

print("\n=== 手持ちの材料:", my_ingredients, "→ 作れそうな料理（和風だけ） ===")
for r in recommend_recipes_from_ingredients(my_ingredients, preferred_genres=["和風"],
                                            top_k=5):
    print(f"{r['score']:.3f}  [{r['genre']}] {r['name']}  材料: {r['ingredients']}")

print("\n=== これも買えば？（和風料理を前提に不足材料を提案） ===")
for e in recommend_extra_ingredients(my_ingredients,
                                     preferred_genres=["和風"],
                                     top_k_recipes=3,
                                     top_k_ingredients=5):
    print(f"{e['score']:.3f}  {e['name']}")
