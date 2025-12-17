from sentence_transformers import util

def recommend(query_index, embeddings, products, top_k=5):
    """
    Returns top_k most similar products (excluding itself)
    """
    scores = util.cos_sim(embeddings[query_index], embeddings)[0]

    ranked = sorted(
        enumerate(scores),
        key=lambda x: float(x[1]),
        reverse=True
    )

    results = []
    for idx, score in ranked:
        if idx == query_index:
            continue
        results.append({
            "id": products[idx]["id"],
            "name": products[idx]["name"],
            "score": round(float(score), 3)
        })
        if len(results) == top_k:
            break

    return results
