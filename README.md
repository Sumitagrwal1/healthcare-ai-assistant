# Healthcare AI Assistant (RAG + LLM)

Mindbowser hackathon prototype using the preferred stack from the assignment brief.

## Architecture

```
User question
    → LangChain router (TOOL vs RAG)
        → LangChain tool: check_available_slots (mock)
        → OR FAISS RAG pipeline
              → HuggingFace embeddings (MiniLM)
              → retrieve top chunks from FAISS
              → LLM answers only from context
              → answer + sources + confidence
```

## Stack (PDF preferred)

| Part | Choice |
|------|--------|
| API | FastAPI |
| Vector DB | FAISS |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Agent / tools | LangChain (`check_available_slots` tool + router) |
| LLM | Groq API |

## Prompt strategy

```
You are a healthcare policy assistant.
Answer ONLY using the CONTEXT below.
If the answer is not in CONTEXT, reply exactly: I could not find this information in the provided documents.
Do not guess. Do not give medical diagnosis or unsafe medical advice.
Keep the tone clear and professional.
```

## Setup

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`.env` is included so evaluators can run the demo without creating their own key. After evaluation, rotate the Groq API key.

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

- UI: http://localhost:8000
- API docs: http://localhost:8000/docs

### Docker

```bash
docker compose up --build
```

## API examples

```bash
curl -X POST http://localhost:8000/ingest
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"Can a patient request a medication refill through telehealth?\"}"
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"Can I book a cardiology appointment for Monday?\"}"
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"What is the clinic Wi-Fi password?\"}"
```

## Sample Q&A

**Q:** Can a patient request a medication refill through telehealth?  
**A:** Yes, when already prescribed and no in-person evaluation is required.  
**Sources:** `telehealth_policy.txt`

**Q:** Can I book a cardiology appointment for Monday?  
**A:** Mock slots via LangChain tool `check_available_slots`.

**Q:** What is the clinic Wi-Fi password?  
**A:** I could not find this information in the provided documents.

## Dataset

Synthetic healthcare policies in `/data` (no real PHI).

## Project structure

```
healthcare-ai-assistant/
  app/
    main.py
    rag.py
    embeddings.py
    llm.py
    agent.py
    config.py
    static/
  data/
  vector_store/
  tests/
  requirements.txt
  Dockerfile
  docker-compose.yml
  README.md
```

## Limitations and future improvements

- Mock appointment tool is not connected to a real EHR/scheduler.
- Add auth, audit logs, PHI redaction, and faithfulness evaluation.
- For multi-tenant production, move FAISS to a managed vector DB (e.g. Chroma/Pinecone cloud).
