from langchain_huggingface import HuggingFaceEmbeddings
from app.config import EMBEDDING_MODEL

_emb = None

def get_embeddings():
    global _emb
    if _emb is None:
        _emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})
    return _emb
