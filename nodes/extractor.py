import json
import re
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=4096)

# ── Prompt ───────────────────────────────────────────────────────────────────
EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial data extraction expert.
Extract ALL transactions from the bank statement text below.

Return ONLY a valid JSON array. No explanation, no markdown, no code fences.
Do NOT truncate. Every transaction must be included.

Each transaction must have exactly these fields:
- date        (string, format: YYYY-MM-DD)
- description (string, merchant or memo text)
- amount      (float, negative=debit, positive=credit)
- balance     (float or null if not shown)
- type        (string: "debit" or "credit")

Example:
[
  {{"date": "2024-01-15", "description": "AMAZON PURCHASE", "amount": -49.99, "balance": 1200.00, "type": "debit"}},
  {{"date": "2024-01-16", "description": "SALARY DEPOSIT",  "amount": 3000.00, "balance": 4200.00, "type": "credit"}}
]"""),
    ("human", "{text}")
])

extraction_chain = EXTRACTION_PROMPT | llm


def _parse_response(raw: str) -> list[dict]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"```$", "", raw).strip()
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    if not raw.startswith("["):
        raw = "[" + raw
    if not raw.endswith("]"):
        raw = raw + "]"
    return json.loads(raw)


def _chunk_text(text: str, max_chars: int = 3000) -> list[str]:
    """Split text into chunks at newline boundaries."""
    lines  = text.split("\n")
    chunks = []
    current = ""
    for line in lines:
        if len(current) + len(line) > max_chars:
            if current.strip():
                chunks.append(current.strip())
            current = line + "\n"
        else:
            current += line + "\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


def extract_transactions(redacted_text: str) -> dict:
    """
    Extract transactions chunk by chunk to avoid token limit cutoffs.
    Merges and deduplicates results across all chunks.
    """
    chunks = _chunk_text(redacted_text, max_chars=3000)
    all_transactions = []
    seen = set()

    for i, chunk in enumerate(chunks):
        try:
            response = extraction_chain.invoke({"text": chunk})
            raw      = response.content
            txns     = _parse_response(raw)

            for t in txns:
                t["amount"]  = float(t.get("amount", 0))
                t["balance"] = float(t["balance"]) if t.get("balance") is not None else None
                t["date"]    = str(t.get("date", ""))

                # Deduplicate by date + description + amount
                key = (t["date"], t["description"], t["amount"])
                if key not in seen:
                    seen.add(key)
                    all_transactions.append(t)

        except Exception as e:
            # Skip chunks that fail — don't crash the whole pipeline
            print(f"[extractor] chunk {i+1}/{len(chunks)} failed: {e}")
            continue

    if not all_transactions:
        raise ValueError("No transactions could be extracted from any chunk.")

    df = pd.DataFrame(all_transactions)
    return {
        "transactions": all_transactions,
        "dataframe":    df,
        "count":        len(all_transactions),
    }