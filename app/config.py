import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR, VECTOR_DIR = ROOT / "data", ROOT / "vector_store"
CHUNK_SIZE, CHUNK_OVERLAP = int(os.getenv("CHUNK_SIZE", 500)), int(os.getenv("CHUNK_OVERLAP", 80))
TOP_K, MIN_SCORE = int(os.getenv("TOP_K", 3)), float(os.getenv("MIN_SCORE", 0.2))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
PROMPT = """You are a healthcare policy assistant.
Answer ONLY using the CONTEXT below.
If the answer is not in CONTEXT, reply exactly: I could not find this information in the provided documents.
Do not guess. Do not give medical diagnosis or unsafe medical advice.
Keep the tone clear and professional.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
