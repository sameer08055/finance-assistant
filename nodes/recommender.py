import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a personal finance advisor with access to the user's \
bank statement analysis.

You have been given:
- A financial summary (income, expenses, net cashflow, category breakdown)
- A list of detected anomalies (unusual or duplicate transactions)
- The user's question or request

Your job:
1. Answer the user's question accurately using the financial data provided
2. Be specific — reference actual numbers, categories, and dates from the data
3. Flag any anomalies that are relevant to the question
4. Give 2-3 actionable recommendations when appropriate
5. Keep responses concise and friendly

Format numbers as currency where relevant (e.g. $1,234.56).
Never make up transactions or figures not present in the data.
If you cannot answer from the data provided, say so clearly."""

# ── Prompt Template ───────────────────────────────────────────────────────────
RECOMMENDER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """FINANCIAL SUMMARY:
{summary}

ANOMALIES DETECTED:
{anomalies}

RELEVANT TRANSACTIONS (semantic search results):
{relevant_transactions}

USER QUESTION:
{question}"""),
])

# ── Chain ─────────────────────────────────────────────────────────────────────
recommender_chain = RECOMMENDER_PROMPT | llm


def _format_summary(summary: dict) -> str:
    """Render summary dict as readable text for the prompt."""
    lines = [
        f"- Total Income:      ${summary.get('total_income', 0):,.2f}",
        f"- Total Expenses:    ${summary.get('total_expenses', 0):,.2f}",
        f"- Net Cashflow:      ${summary.get('net_cashflow', 0):,.2f}",
        f"- Transaction Count: {summary.get('transaction_count', 0)}",
        f"- Avg Expense:       ${summary.get('avg_expense', 0):,.2f}",
        f"- Largest Expense:   ${summary.get('largest_expense', 0):,.2f}",
        "",
        "Spending by Category:",
    ]
    for cat in summary.get("spending_by_category", []):
        lines.append(f"  • {cat['category']}: ${cat['total']:,.2f} ({cat['count']} transactions)")

    lines.append("\nMonthly Breakdown:")
    for month in summary.get("monthly_breakdown", []):
        lines.append(
            f"  • {month['month']}: "
            f"Income ${month['income']:,.2f} | "
            f"Expenses ${month['expenses']:,.2f}"
        )
    return "\n".join(lines)


def _format_anomalies(anomalies: list[dict]) -> str:
    """Render anomalies as readable text for the prompt."""
    if not anomalies:
        return "No anomalies detected."
    lines = []
    for a in anomalies:
        lines.append(
            f"  • [{a['method'].upper()}] {a['date']} | "
            f"{a['description']} | "
            f"${abs(a['amount']):,.2f} — {a['reason']}"
        )
    return "\n".join(lines)


def _format_relevant(relevant: list[dict]) -> str:
    """Render semantic search results as readable text for the prompt."""
    if not relevant:
        return "No specific transactions retrieved."
    lines = []
    for r in relevant:
        lines.append(
            f"  • {r.get('date', '')} | "
            f"{r.get('description', '')} | "
            f"${abs(r.get('amount', 0)):,.2f} | "
            f"{r.get('category', '')} "
            f"(score: {r.get('similarity_score', 0):.2f})"
        )
    return "\n".join(lines)


def get_recommendation(
    question: str,
    summary: dict,
    anomalies: list[dict],
    relevant_transactions: list[dict],
    chat_history: list | None = None,
) -> str:
    """
    Generate a financial recommendation or answer based on:
    - User question
    - Financial summary
    - Detected anomalies
    - Semantically relevant transactions

    Returns the LLM response as a string.
    """
    response = recommender_chain.invoke({
        "question":             question,
        "summary":              _format_summary(summary),
        "anomalies":            _format_anomalies(anomalies),
        "relevant_transactions": _format_relevant(relevant_transactions),
    })

    return response.content