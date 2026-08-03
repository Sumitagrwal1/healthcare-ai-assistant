from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, MIN_SCORE, PROMPT, TOP_K, VECTOR_DIR
from app.embeddings import get_embeddings
from app.llm import get_llm

NOT_FOUND = "I could not find this information in the provided documents."
INDEX = VECTOR_DIR / "faiss_index"

def ingest():
    docs = DirectoryLoader(str(DATA_DIR), glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}).load()
    if not docs:
        raise FileNotFoundError("No .txt files in /data")
    chunks = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_documents(docs)
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    FAISS.from_documents(chunks, get_embeddings()).save_local(str(INDEX))
    return {"files": len(docs), "chunks": len(chunks)}

def _store():
    if not INDEX.exists():
        raise FileNotFoundError("Vector store missing. Call POST /ingest first.")
    return FAISS.load_local(str(INDEX), get_embeddings(), allow_dangerous_deserialization=True)

def retrieve(q):
    return [(d, s) for d, s in _store().similarity_search_with_relevance_scores(q, k=TOP_K) if s >= MIN_SCORE]

def ask(q):
    pairs = retrieve(q)
    if not pairs:
        return {"answer": NOT_FOUND, "sources": [], "confidence": "low"}
    docs = [d for d, _ in pairs]
    ctx = "\n\n".join(f"[{Path(d.metadata.get('source','doc')).name}] {d.page_content}" for d in docs)
    ans = get_llm().invoke(PROMPT.format(context=ctx, question=q)).content.strip()
    if NOT_FOUND.lower() in ans.lower():
        return {"answer": NOT_FOUND, "sources": [], "confidence": "low"}
    top = pairs[0][1]
    conf = "high" if top >= 0.5 else "medium" if top >= 0.22 else "low"
    src = [{"document": Path(d.metadata.get("source","doc")).name, "chunk": d.page_content[:220]} for d, _ in pairs]
    return {"answer": ans, "sources": src, "confidence": conf}
