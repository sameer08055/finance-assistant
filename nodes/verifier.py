import json
import re
from datetime import datetime
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

CORRECTIONS_LOG = Path(__file__).parent.parent / "eval" / "corrections.log"

_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=4096)

_CORRECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial data extraction expert.
The transactions below failed validation. Return corrected versions.

Return ONLY a valid JSON array. No explanation, no markdown, no code fences.
Preserve the same order as the input.

Each transaction must have exactly these fields:
- date        (string, format: YYYY-MM-DD)
- description (string, non-empty merchant or memo text)
- amount      (float, negative=debit, positive=credit)
- balance     (float or null if not shown)
- type        (string: "debit" or "credit")
"""),
    ("human", "The following transactions failed validation. Re-extract them carefully:\n{failed_rows}"),
])

_correction_chain = _CORRECTION_PROMPT | _llm


def _log_correction(original: dict, corrected: dict, reason: str) -> None:
    CORRECTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().isoformat() + "Z"
    line = (
        f"{timestamp}"
        f" | original={json.dumps(original, default=str)}"
        f" | corrected={json.dumps(corrected, default=str)}"
        f" | reason={reason}\n"
    )
    with open(CORRECTIONS_LOG, "a") as fh:
        fh.write(line)


def _is_valid_date(value) -> bool:
    if not value or not isinstance(value, str):
        return False
    try:
        datetime.strptime(str(value)[:10], "%Y-%m-%d")
        return True
    except (ValueError, TypeError):
        return False


def _is_valid_float(value) -> bool:
    if value is None:
        return False
    try:
        f = float(value)
        return f == f  # NaN != NaN
    except (TypeError, ValueError):
        return False


def _schema_failures(txn: dict) -> list[str]:
    """Return list of validation failure reasons; empty = valid."""
    reasons = []

    if not _is_valid_date(txn.get("date")):
        reasons.append(f"invalid date: {txn.get('date')!r}")

    if not _is_valid_float(txn.get("amount")):
        reasons.append(f"invalid amount: {txn.get('amount')!r}")

    desc = txn.get("description")
    if not isinstance(desc, str) or not desc.strip():
        reasons.append(f"empty/missing description: {desc!r}")

    balance = txn.get("balance")
    if balance is not None and not _is_valid_float(balance):
        reasons.append(f"invalid balance: {balance!r}")

    txn_type = str(txn.get("type", "")).lower()
    if txn_type not in ("debit", "credit"):
        reasons.append(f"invalid type: {txn.get('type')!r}")

    return reasons


def _balance_failures(transactions: list[dict]) -> list[tuple[int, str]]:
    """
    For each consecutive pair of transactions (sorted by date), verify:
        balance[i] + amount[i+1] ≈ balance[i+1]  (within $0.01)
    Returns list of (original_index, reason) for failing rows.
    """
    indexed = list(enumerate(transactions))
    indexed.sort(key=lambda x: str(x[1].get("date", "")))

    valid_seq = [
        (orig_i, t) for orig_i, t in indexed
        if _is_valid_float(t.get("balance")) and _is_valid_float(t.get("amount"))
    ]

    failures = []
    for j in range(1, len(valid_seq)):
        prev_i, prev_t = valid_seq[j - 1]
        curr_i, curr_t = valid_seq[j]

        expected = float(prev_t["balance"]) + float(curr_t["amount"])
        actual   = float(curr_t["balance"])

        if abs(expected - actual) > 0.01:
            reason = (
                f"balance inconsistency at index {curr_i}: "
                f"prev_balance({prev_t['balance']}) + amount({curr_t['amount']}) "
                f"= {expected:.2f} ≠ actual_balance({actual:.2f})"
            )
            failures.append((curr_i, reason))

    return failures


def _parse_llm_json(raw: str) -> list[dict]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"```$", "", raw).strip()
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    if not raw.startswith("["):
        raw = "[" + raw
    if not raw.endswith("]"):
        raw = raw + "]"
    return json.loads(raw)


def _diff_fields(original: dict, fixed: dict) -> list[dict]:
    """Return [{field, original_value, corrected_value}] for every changed field."""
    diffs = []
    for field in ("date", "description", "amount", "balance", "type"):
        orig_val = original.get(field)
        corr_val = fixed.get(field)
        if str(orig_val) != str(corr_val):
            diffs.append({
                "field":           field,
                "original_value":  orig_val,
                "corrected_value": corr_val,
            })
    return diffs


def _coerce_types(txn: dict) -> dict:
    t = dict(txn)
    try:
        t["amount"] = float(t.get("amount", 0))
    except (TypeError, ValueError):
        pass
    try:
        if t.get("balance") is not None:
            t["balance"] = float(t["balance"])
    except (TypeError, ValueError):
        pass
    t["date"] = str(t.get("date", ""))
    return t


def verify_transactions(transactions: list[dict]) -> dict:
    """
    Validate, check balance consistency, re-prompt LLM for failures,
    log corrections, and stamp every transaction with verified: true/false.

    Returns:
        transactions     : list[dict]  (with verified flag)
        corrections_made : int
        failed_schema    : list[int]   (original indices)
        failed_balance   : list[int]   (original indices)
        corrections      : list[dict]  (per-correction detail for UI)
    """
    if not transactions:
        return {
            "transactions":    [],
            "corrections_made": 0,
            "failed_schema":   [],
            "failed_balance":  [],
            "corrections":     [],
        }

    # ── 1. Schema validation ──────────────────────────────────────────────────
    schema_fail_map: dict[int, list[str]] = {}
    for i, txn in enumerate(transactions):
        reasons = _schema_failures(txn)
        if reasons:
            schema_fail_map[i] = reasons

    # ── 2. Sequential balance consistency ────────────────────────────────────
    bal_failures = _balance_failures(transactions)
    bal_fail_map: dict[int, str] = {i: r for i, r in bal_failures}

    all_failed = set(schema_fail_map) | set(bal_fail_map)

    # ── 3. Re-prompt LLM for failed rows ─────────────────────────────────────
    corrected        = list(transactions)
    corrections_made = 0
    corrections_list: list[dict] = []

    if all_failed:
        sorted_failed = sorted(all_failed)
        payload = [
            {
                "original_index": idx,
                "transaction":    transactions[idx],
                "failures": (
                    schema_fail_map.get(idx, []) +
                    ([bal_fail_map[idx]] if idx in bal_fail_map else [])
                ),
            }
            for idx in sorted_failed
        ]

        try:
            response = _correction_chain.invoke(
                {"failed_rows": json.dumps(payload, indent=2, default=str)}
            )
            new_txns = _parse_llm_json(response.content)

            for j, idx in enumerate(sorted_failed):
                if j >= len(new_txns):
                    break
                original = transactions[idx]
                fixed    = _coerce_types(new_txns[j])
                reasons  = (
                    schema_fail_map.get(idx, []) +
                    ([bal_fail_map[idx]] if idx in bal_fail_map else [])
                )
                _log_correction(original, fixed, "; ".join(reasons))
                corrected[idx] = fixed
                corrections_made += 1
                corrections_list.append({
                    "description":  original.get("description", ""),
                    "date":         str(original.get("date", ""))[:10],
                    "reason":       "; ".join(reasons),
                    "fields":       _diff_fields(original, fixed),
                })

        except Exception as e:
            print(f"[verifier] re-prompt failed: {e}")

    # ── 4. verified flag (re-validate after corrections) ─────────────────────
    for txn in corrected:
        txn["verified"] = len(_schema_failures(txn)) == 0

    return {
        "transactions":    corrected,
        "corrections_made": corrections_made,
        "failed_schema":   list(schema_fail_map),
        "failed_balance":  list(bal_fail_map),
        "corrections":     corrections_list,
    }
