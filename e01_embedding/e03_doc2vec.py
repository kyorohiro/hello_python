from gensim.models.doc2vec import Doc2Vec, TaggedDocument

texts = [
    "今日はカレーを作った",
    "昨日はラーメンを食べた",
    "明日はカレーうどんを食べたい",
]

documents = [
    TaggedDocument(words=text.split(), tags=[f"doc_{i}"])
    for i, text in enumerate(texts)
]

model = Doc2Vec(vector_size=100, min_count=1, workers=4, epochs=40)
model.build_vocab(documents)
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

# 新しい文章を embedding にする
vec = model.infer_vector("カレーのレシピを知りたい".split())
print(vec.shape)  # (100,)

query = "カレーのレシピを知りたい"
query_vec = model.infer_vector(query.split())

similar_docs = model.dv.most_similar([query_vec], topn=3)

for tag, score in similar_docs:
    idx = int(tag.split("_")[1])
    print(f"[{tag}] score={score:.3f}  ->  {texts[idx]}")

# 保存
model.save("doc2vec-demo.model")

# 読み込み
# from gensim.models.doc2vec import Doc2Vec
# model = Doc2Vec.load("doc2vec-demo.model")

