import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss


def build_index(data_path):
    df = pd.read_csv(data_path)
    df["email_text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["email_text"].tolist(), convert_to_numpy=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return model, index, df


def search(query, model, index, df, top_k=3):
    emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, ids = index.search(emb, top_k)

    results = []
    for idx, score in zip(ids[0], scores[0]):
        results.append({
            "email": df.iloc[idx]["email_text"],
            "tag": df.iloc[idx]["tag"],
            "score": float(score)
        })

    return results


if __name__ == "__main__":
    model, index, df = build_index("../data/large_dataset.csv")
    res = search("unable to assign emails automatically", model, index, df)
    print(res)
