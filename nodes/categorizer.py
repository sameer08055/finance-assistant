import json
import re
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ── Categories ────────────────────────────────────────────────────────────────
CATEGORIES = [
    "Food & Dining",
    "Groceries",
    "Shopping",
    "Transportation",
    "Travel",
    "Entertainment",
    "Health & Medical",
    "Utilities",
    "Rent & Housing",
    "Subscriptions",
    "Income & Salary",
    "Transfers",
    "ATM & Cash",
    "Insurance",
    "Education",
    "Other",
]

# ── Prompt ────────────────────────────────────────────────────────────────────
CATEGORIZATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial transaction categorizer.
Assign each transaction exactly one category from this list:
{categories}

Return ONLY a valid JSON array. No explanation, no markdown, no code fences.
Each item must have:
- description (string, same as input)
- category    (string, must be from the list above)
- confidence  (float, 0.0 to 1.0)

Example:
[
  {{"description": "WHOLE FOODS MARKET", "category": "Groceries",      "confidence": 0.97}},
  {{"description": "NETFLIX.COM",        "category": "Subscriptions",  "confidence": 0.99}},
  {{"description": "UBER TRIP",          "category": "Transportation", "confidence": 0.95}}
]"""),
    ("human", "{transactions}")
])

# ── Chain ─────────────────────────────────────────────────────────────────────
categorization_chain = CATEGORIZATION_PROMPT | llm


def _parse_response(raw: str) -> list[dict]:
    raw = raw.strip()
    # Strip markdown fences
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"```$", "", raw).strip()
    # Remove trailing commas before ] or }
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    # If it doesn't start with [ wrap it
    if not raw.startswith("["):
        raw = "[" + raw
    if not raw.endswith("]"):
        raw = raw + "]"
    return json.loads(raw)


def _batch(lst: list, size: int):
    """Yield successive chunks of `size` from lst."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def categorize_transactions(transactions: list[dict]) -> dict:
    """
    Categorize transactions in batches of 20 to stay within token limits.
    Returns: {transactions: list[dict], dataframe: pd.DataFrame}
    """
    # Build minimal input to reduce tokens
    slim = [{"description": t["description"]} for t in transactions]

    category_map: dict[str, dict] = {}

    for chunk in _batch(slim, 20):
        response = categorization_chain.invoke({
            "categories":    "\n".join(f"- {c}" for c in CATEGORIES),
            "transactions":  json.dumps(chunk, indent=2),
        })
        raw = response.content

        try:
            results = _parse_response(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}\n\nRaw:\n{raw}")

        for r in results:
            category_map[r["description"]] = {
                "category":   r.get("category", "Other"),
                "confidence": float(r.get("confidence", 0.0)),
            }

    # Merge categories back into transactions
    categorized = []
    for t in transactions:
        match = category_map.get(t["description"], {"category": "Other", "confidence": 0.0})
        categorized.append({**t, **match})

    df = pd.DataFrame(categorized)

    return {
        "transactions": categorized,
        "dataframe":    df,
    }