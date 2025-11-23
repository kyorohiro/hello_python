import gensim.downloader as api
from gensim.similarities.annoy import AnnoyIndexer

def main() -> None:
    print("Loading GloVe model (100d)...")
    model = api.load("glove-wiki-gigaword-100")
    print("vocab size:", len(model))

    # 50 trees くらいで ANN Index を作る
    annoy_index = AnnoyIndexer(model, num_trees=50)
    # memo アルゴリズム Faiss / Annoy / hnswlib
    # memo ベクトルDB側（Qdrant / Milvus / Weaviate / pgvector）
    # など DBを使う方法もある

    # ベクトル演算: king - man + woman
    king = model["king"]
    man = model["man"]
    woman = model["woman"]

    vec = king - man + woman

    print("\n=== king - man + woman に近い単語 Top 10 ===")
    for word, score in model.most_similar([vec], topn=10, indexer=annoy_index):
        print(f"{word:10s}  {score:.4f}")

    print("\n=== いくつかの類似度 ===")
    pairs = [
        ("king", "queen"),
        ("king", "man"),
        ("king", "apple"),
        ("tokyo", "japan"),
        ("tokyo", "france"),
    ]
    for w1, w2 in pairs:
        sim = model.similarity(w1, w2)
        print(f"{w1:10s} vs {w2:10s} → {sim:.4f}")


if __name__ == "__main__":
    main()
