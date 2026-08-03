import logging, sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None

from app.agent import handle
from app.config import GROQ_MODEL
from app.rag import ingest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("healthcare-ai")
STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Healthcare AI Assistant", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

class AskRequest(BaseModel):
    question: str = Field(min_length=3)

@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")

@app.get("/health")
def health():
    return {"status": "ok", "llm_provider": "groq", "model": GROQ_MODEL}

@app.post("/ingest")
def ingest_docs():
    try:
        return {"status": "success", **ingest()}
    except Exception as e:
        log.exception("Ingest failed")
        raise HTTPException(500, str(e)) from e

@app.post("/ask")
def ask_question(body: AskRequest):
    try:
        return handle(body.question.strip())
    except FileNotFoundError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        log.exception("Ask failed")
        raise HTTPException(500, str(e)) from e
