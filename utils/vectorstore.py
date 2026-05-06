import faiss
import numpy as np
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# ── Embeddings ────────────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


def _transaction_to_document(t: dict) -> Document:
    """
    Convert a transaction dict into a LangChain Document.
    Page content is a natural language description for semantic search.
    Metadata stores all structured fields.
    """
    content = (
        f"{t['date']} | {t['description']} | "
        f"{'spent' if t['amount'] < 0 else 'received'} "
        f"${abs(t['amount']):.2f} | category: {t.get('category', 'Unknown')}"
    )
    return Document(
        page_content=content,
        metadata={
            "date":        t.get("date", ""),
            "description": t.get("description", ""),
            "amount":      t.get("amount", 0.0),
            "type":        t.get("type", ""),
            "category":    t.get("category", "Other"),
            "confidence":  t.get("confidence", 0.0),
            "balance":     t.get("balance"),
        }
    )


def build_vectorstore(transactions: list[dict]) -> FAISS:
    """
    Build a FAISS vectorstore from a list of categorized transaction dicts.
    Returns a LangChain FAISS vectorstore instance.
    """
    if not transactions:
        raise ValueError("Cannot build vectorstore from empty transactions list.")

    docs = [_transaction_to_document(t) for t in transactions]
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


def semantic_search(
    vectorstore: FAISS,
    query: str,
    k: int = 5,
    filter_category: str | None = None,
) -> list[dict]:
    """
    Search for transactions semantically similar to the query.
    Optionally filter by category.
    Returns a list of metadata dicts with a similarity score.
    """
    if filter_category:
        results = vectorstore.similarity_search_with_score(
            query, k=k * 2  # fetch more, then filter
        )
        results = [
            (doc, score) for doc, score in results
            if doc.metadata.get("category") == filter_category
        ][:k]
    else:
        results = vectorstore.similarity_search_with_score(query, k=k)

    return [
        {**doc.metadata, "similarity_score": round(float(score), 4)}
        for doc, score in results
    ]


def save_vectorstore(vectorstore: FAISS, path: str = "faiss_index") -> None:
    """Persist the FAISS index to disk."""
    vectorstore.save_local(path)


def load_vectorstore(path: str = "faiss_index") -> FAISS:
    """Load a persisted FAISS index from disk."""
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)