# ================================
# インストール
# ================================
# !pip install -q sentence-transformers

from sentence_transformers import SentenceTransformer
import numpy as np


# ================================
# レシピデータ（料理名 + 材料[]）
# ================================
recipes = [
    {
        "name": "寄せ鍋",
        "ingredients": ["白菜", "長ねぎ", "豆腐", "豚肉", "しめじ", "しらたき"]
    },
    {
        "name": "キムチ鍋",
        "ingredients": ["白菜", "豚肉", "豆腐", "キムチの素", "長ねぎ"]
    },
    {
        "name": "水炊き",
        "ingredients": ["鶏肉", "白菜", "豆腐", "長ねぎ", "昆布"]
    },
    {
        "name": "味噌汁",
        "ingredients": ["味噌", "豆腐", "わかめ", "だし", "長ねぎ"]
    },
    {
        "name": "豚汁",
        "ingredients": ["豚肉", "大根", "にんじん", "味噌", "ごぼう", "長ねぎ"]
    },
]


# ================================
# embedding モデル
# ================================
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def recipe_to_text(recipe):
    ing = "、".join(recipe["ingredients"])
    return f"{recipe['name']}。材料: {ing}"

# 全レシピをテキスト化 & embedding
recipe_texts = [recipe_to_text(r) for r in recipes]
recipe_embeddings = model.encode(recipe_texts, normalize_embeddings=True)


# ================================
# ① 材料 → 作れる料理の推薦
# ================================
def recommend_recipes_from_ingredients(ingredients_list, top_k=5):
    """
    ユーザーが持っている材料から作れそうな料理を推薦
    """
    query_text = " ".join(ingredients_list)
    query_emb = model.encode([query_text], normalize_embeddings=True)[0]

    scores = recipe_embeddings @ query_emb
    idxs = np.argsort(scores)[::-1][:top_k]

    return [(recipes[i]["name"], float(scores[i])) for i in idxs]


# ================================
# ② 料理名 → 足りない材料の推薦
# ================================
def recommend_missing_ingredients(dish_name, user_ingredients, top_k=5):
    """
    料理名から、その料理に必要な材料のうち、
    ユーザーが持っていない材料を推薦
    """
    # 入力料理名を embedding 化
    dish_emb = model.encode([dish_name], normalize_embeddings=True)[0]

    # 一番近いレシピを探す
    scores = recipe_embeddings @ dish_emb
    best_idx = int(np.argmax(scores))
    best_recipe = recipes[best_idx]

    # 足りない材料
    missing = [
        ing for ing in best_recipe["ingredients"] 
        if ing not in user_ingredients
    ]

    return missing[:top_k]


# ================================
# ③ 料理名 → 似てる料理の推薦
# ================================
def recommend_similar_recipes(dish_name, top_k=5):
    dish_emb = model.encode([dish_name], normalize_embeddings=True)[0]
    scores = recipe_embeddings @ dish_emb
    idxs = np.argsort(scores)[::-1][:top_k]

    return [(recipes[i]["name"], float(scores[i])) for i in idxs]



# ================================
# ★ 動作確認 ★
# ================================

print("=== 材料 → 作れる料理 ===")
print(recommend_recipes_from_ingredients(["白菜", "豆腐", "鶏肉"], top_k=3))

print("\n=== 料理名 → 足りない材料 ===")
print(recommend_missing_ingredients("寄せ鍋", ["白菜", "豚肉"], top_k=5))

print("\n=== 料理名 → 似ている料理 ===")
print(recommend_similar_recipes("寄せ鍋", top_k=3))
