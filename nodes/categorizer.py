import json
import logging
import re
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ── "Other" logger ────────────────────────────────────────────────────────────
_other_logger = logging.getLogger("categorizer.other")
if not _other_logger.handlers:
    _handler = logging.FileHandler("other_transactions.log")
    _handler.setFormatter(logging.Formatter("%(asctime)s\t%(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    _other_logger.addHandler(_handler)
    _other_logger.setLevel(logging.INFO)
    _other_logger.propagate = False

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

Guidelines (use "Other" ONLY as a last resort — most transactions fit a specific category):
- Food & Dining: restaurants, cafes, fast food, bars, coffee shops, food delivery (DoorDash, Uber Eats, Grubhub)
- Groceries: supermarkets, grocery stores (Whole Foods, Trader Joe's, Kroger, Safeway, Costco, Walmart, Target food purchases)
- Shopping: retail stores, Amazon, online shopping, clothing, electronics, home goods
- Transportation: Uber, Lyft, taxi, gas stations, parking, tolls, public transit, car services
- Travel: airlines, hotels, Airbnb, car rental, travel agencies
- Entertainment: movies, concerts, sports, streaming (Netflix, Spotify, Hulu, Disney+), gaming, hobbies
- Health & Medical: pharmacies, doctors, hospitals, dentist, gyms, fitness, CVS/Walgreens (medical)
- Utilities: electric, gas, water, internet, phone, cable, trash
- Rent & Housing: rent, mortgage, HOA, property management
- Subscriptions: recurring software, SaaS, membership fees, annual/monthly services
- Income & Salary: direct deposit, payroll, employer payments, tax refunds
- Transfers: bank transfers, Venmo, Zelle, PayPal transfers between accounts, wire transfers
- ATM & Cash: ATM withdrawals, cash advances
- Insurance: health, auto, home, life, renters insurance premiums
- Education: tuition, textbooks, online courses, student fees

When a transaction could fit multiple categories, pick the most specific one.
"Other" is reserved for transactions that genuinely do not fit any category above.

Return ONLY a valid JSON array. No explanation, no markdown, no code fences.
Each item must have:
- description (string, same as input)
- category    (string, must be from the list above)
- confidence  (float, 0.0 to 1.0)

Example:
[
  {{"description": "WHOLE FOODS MARKET", "category": "Groceries",      "confidence": 0.97}},
  {{"description": "NETFLIX.COM",        "category": "Subscriptions",  "confidence": 0.99}},
  {{"description": "UBER TRIP",          "category": "Transportation", "confidence": 0.95}},
  {{"description": "CVS PHARMACY",       "category": "Health & Medical","confidence": 0.88}},
  {{"description": "SHELL OIL",          "category": "Transportation", "confidence": 0.92}}
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
        result = {**t, **match}
        if result["category"] == "Other":
            _other_logger.info(
                "desc=%r\tamount=%s\tdate=%s\tconfidence=%.2f",
                t.get("description", ""),
                t.get("amount", ""),
                t.get("date", ""),
                match.get("confidence", 0.0),
            )
        categorized.append(result)

    df = pd.DataFrame(categorized)

    return {
        "transactions": categorized,
        "dataframe":    df,
    }