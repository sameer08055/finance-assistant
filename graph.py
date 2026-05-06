from typing import TypedDict, Annotated
import operator
import pandas as pd
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from nodes.pii         import redact_pii
from nodes.extractor   import extract_transactions
from nodes.categorizer import categorize_transactions
from nodes.analyzer    import (
    load_transactions_to_sql,
    get_financial_summary,
    detect_anomalies,
)
from nodes.recommender import get_recommendation
from utils.vectorstore import build_vectorstore, semantic_search

load_dotenv()


# ── Sanitizer ─────────────────────────────────────────────────────────────────
def _sanitize(obj):
    """Recursively convert numpy/pandas types to native Python types."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize(i) for i in obj]
    elif hasattr(obj, "item"):      # numpy.float64, numpy.int64, etc.
        return obj.item()
    elif obj != obj:                # NaN
        return None
    return obj


# ── State ─────────────────────────────────────────────────────────────────────
class FinanceState(TypedDict):
    # Input
    raw_text:        str
    question:        str

    # Pipeline outputs
    redacted_text:   str
    pii_findings:    list[dict]
    transactions:    list[dict]
    summary:         dict
    anomalies:       list[dict]
    relevant_txns:   list[dict]
    answer:          str

    # Conversation memory
    chat_history:    Annotated[list, operator.add]

    # Control
    error:           str
    stage:           str


# ── Nodes ─────────────────────────────────────────────────────────────────────
def node_redact(state: FinanceState) -> dict:
    try:
        result = redact_pii(state["raw_text"])
        return {
            "redacted_text": result["redacted_text"],
            "pii_findings":  _sanitize(result["findings"]),
            "stage":         "redacted",
        }
    except Exception as e:
        return {"error": str(e), "stage": "error"}


def node_extract(state: FinanceState) -> dict:
    try:
        result = extract_transactions(state["redacted_text"])
        return {
            "transactions": _sanitize(result["transactions"]),
            "stage":        "extracted",
        }
    except Exception as e:
        return {"error": str(e), "stage": "error"}


def node_categorize(state: FinanceState) -> dict:
    try:
        result = categorize_transactions(state["transactions"])
        return {
            "transactions": _sanitize(result["transactions"]),
            "stage":        "categorized",
        }
    except Exception as e:
        return {"error": str(e), "stage": "error"}


def node_analyze(state: FinanceState) -> dict:
    try:
        df        = pd.DataFrame(state["transactions"])
        load_transactions_to_sql(df)
        summary   = _sanitize(get_financial_summary())
        anomalies = _sanitize(detect_anomalies(df))
        return {
            "summary":   summary,
            "anomalies": anomalies,
            "stage":     "analyzed",
        }
    except Exception as e:
        return {"error": str(e), "stage": "error"}


def node_vectorstore(state: FinanceState) -> dict:
    try:
        vectorstore = build_vectorstore(state["transactions"])
        relevant    = _sanitize(semantic_search(vectorstore, state["question"], k=5))
        return {
            "relevant_txns": relevant,
            "stage":         "vectorstore_ready",
        }
    except Exception as e:
        return {"error": str(e), "stage": "error"}


def node_recommend(state: FinanceState) -> dict:
    try:
        answer = get_recommendation(
            question=             state["question"],
            summary=              state["summary"],
            anomalies=            state["anomalies"],
            relevant_transactions=state["relevant_txns"],
            chat_history=         state.get("chat_history", []),
        )
        new_history = [
            {"role": "user",      "content": state["question"]},
            {"role": "assistant", "content": answer},
        ]
        return {
            "answer":       answer,
            "chat_history": new_history,
            "stage":        "done",
        }
    except Exception as e:
        return {"error": str(e), "stage": "error"}


# ── Routing ───────────────────────────────────────────────────────────────────
def should_continue(state: FinanceState) -> str:
    if state.get("error"):
        return "error"
    return "continue"


# ── Graph assembly ────────────────────────────────────────────────────────────
def build_graph():
    memory = MemorySaver()
    graph  = StateGraph(FinanceState)

    graph.add_node("redact",      node_redact)
    graph.add_node("extract",     node_extract)
    graph.add_node("categorize",  node_categorize)
    graph.add_node("analyze",     node_analyze)
    graph.add_node("vectorstore", node_vectorstore)
    graph.add_node("recommend",   node_recommend)

    graph.set_entry_point("redact")

    for src, dst in [
        ("redact",      "extract"),
        ("extract",     "categorize"),
        ("categorize",  "analyze"),
        ("analyze",     "vectorstore"),
        ("vectorstore", "recommend"),
    ]:
        graph.add_conditional_edges(
            src,
            should_continue,
            {"continue": dst, "error": END},
        )

    graph.add_edge("recommend", END)

    return graph.compile(checkpointer=memory)


# ── Singleton ─────────────────────────────────────────────────────────────────
finance_graph = build_graph()


# ── Public runners ────────────────────────────────────────────────────────────
def run_pipeline(
    raw_text:  str,
    question:  str,
    thread_id: str = "default",
) -> FinanceState:
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: FinanceState = {
        "raw_text":      raw_text,
        "question":      question,
        "redacted_text": "",
        "pii_findings":  [],
        "transactions":  [],
        "summary":       {},
        "anomalies":     [],
        "relevant_txns": [],
        "answer":        "",
        "chat_history":  [],
        "error":         "",
        "stage":         "start",
    }
    return finance_graph.invoke(initial_state, config=config)


def run_followup(
    question:  str,
    thread_id: str = "default",
) -> str:
    config  = {"configurable": {"thread_id": thread_id}}
    current = finance_graph.get_state(config).values

    if not current.get("transactions"):
        return "No statement loaded yet. Please upload a PDF first."

    updated = {**current, "question": question}
    result  = finance_graph.invoke(updated, config=config)
    return result.get("answer", "Could not generate a response.")