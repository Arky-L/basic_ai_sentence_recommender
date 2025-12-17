from sentence_transformers import SentenceTransformer

# Load once (fast & free)
_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts):
    """
    Convert a list of strings into normalized semantic embeddings
    """
    return _model.encode(texts, normalize_embeddings=True)
