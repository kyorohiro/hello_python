import gensim.downloader as api


def main() -> None:
    print("Loading GloVe model (100d)...")
    model = api.load("glove-wiki-gigaword-100")
    print("vocab size:", len(model))

    # ベクトル演算: king - man + woman
    king = model["king"]
    man = model["man"]
    woman = model["woman"]

    vec = king - man + woman

    print("\n=== king - man + woman に近い単語 Top 10 ===")
    for word, score in model.similar_by_vector(vec, topn=10):
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
