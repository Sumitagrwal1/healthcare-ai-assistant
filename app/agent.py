from datetime import date, timedelta
from langchain.tools import tool
from app.rag import ask as rag_ask

SLOTS = {
    "cardiology": ["09:00 AM", "11:30 AM", "02:00 PM"],
    "dermatology": ["10:00 AM", "01:00 PM", "03:30 PM"],
    "general": ["08:30 AM", "12:00 PM", "04:00 PM"],
}

@tool
def check_available_slots(department: str, date: str) -> str:
    """Return mock appointment slots for a hospital department on a date (YYYY-MM-DD)."""
    dept = department.lower().strip()
    return f"I can check mock appointment availability. Available {dept} slots on {date}: {', '.join(SLOTS.get(dept, SLOTS['general']))}."

def handle(q: str):
    ql = q.lower()
    if any(w in ql for w in ("book", "schedule", "available slot", "available slots", "appointment for", "slots for")):
        dept = next((d for d in SLOTS if d in ql), "general")
        today = date.today()
        day = (today + timedelta(days=((0 - today.weekday()) % 7 or 7))).isoformat() if "monday" in ql else (today + timedelta(days=1)).isoformat()
        ans = check_available_slots.invoke({"department": dept, "date": day})
        return {"answer": ans, "sources": [{"document": "mock_appointment_tool", "chunk": f"{dept},{day}"}], "confidence": "high", "route": "appointment_tool"}
    out = rag_ask(q)
    out["route"] = "rag"
    return out
