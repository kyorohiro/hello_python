# ==============================
# 1. インストール（Colab用）
# ==============================
#!pip install -U sentence-transformers
#
# ==============================
# 2. モデル読み込み
# ==============================
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader

# ベースモデル（好きなやつに変えてOK）
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ==============================
# 3. 学習データ定義（ここを自分のデータに差し替える）
#    label が 0.0〜1.0 の類似度
# ==============================
train_samples = [
    # ほぼ同じ意味 → 1.0 に近く
    InputExample(texts=["低脂肪牛乳 1L", "ローファットミルク 1L"], label=0.95),

    # 同じカテゴリの兄弟（そこそこ近い）
    InputExample(texts=["低脂肪牛乳 1L", "成分無調整 牛乳 1L"], label=0.8),

    # 親子っぽい関係（抽象と具体）
    InputExample(texts=["牛乳", "低脂肪牛乳 1L"], label=0.7),

    # 横断的な関連（朝食セット）
    InputExample(texts=["低脂肪牛乳 1L", "グラノーラ シリアル"], label=0.5),

    # あまり関係ない（ほぼ Negative）
    InputExample(texts=["低脂肪牛乳 1L", "単3形アルカリ乾電池 4本"], label=0.0),
]

# ==============================
# 4. DataLoader & Loss 定義
# ==============================
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=4)

# CosineSimilarityLoss：label(0〜1) を使って類似度を学習
train_loss = losses.CosineSimilarityLoss(model)

# ==============================
# 5. 学習実行（お試しなので 1 epoch）
# ==============================
num_epochs = 1
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    show_progress_bar=True,
    output_path="./my-cosine-model"  # 学習後モデルの保存先
)

# ==============================
# 6. 学習済みモデルで類似度を確認
# ==============================
# 再ロード（別プロセス/別ノートでも同じように使える）
fine_tuned_model = SentenceTransformer("./my-cosine-model")

def show_sim(a: str, b: str):
    emb1 = fine_tuned_model.encode(a, convert_to_tensor=True, normalize_embeddings=True)
    emb2 = fine_tuned_model.encode(b, convert_to_tensor=True, normalize_embeddings=True)
    sim = util.cos_sim(emb1, emb2).item()
    print(f"「{a}」 vs 「{b}」 -> cos類似度: {sim:.3f}")

print("\n=== 類似度チェック ===")
show_sim("低脂肪牛乳 1L", "ローファットミルク 1L")        # 高いはず
show_sim("低脂肪牛乳 1L", "成分無調整 牛乳 1L")           # そこそこ高い
show_sim("低脂肪牛乳 1L", "グラノーラ シリアル")          # 中くらい
show_sim("低脂肪牛乳 1L", "単3形アルカリ乾電池 4本")       # かなり低いはず
