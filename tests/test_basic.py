from app.agent import check_available_slots, handle
from app.rag import ingest, ask

def test_tool():
    assert "Available cardiology" in check_available_slots.invoke({"department": "cardiology", "date": "2026-08-03"})

def test_rag():
    ingest()
    assert ask("Can a patient request a medication refill through telehealth?")["sources"]
    assert "could not find" in ask("What is the clinic Wi-Fi password?")["answer"].lower()
