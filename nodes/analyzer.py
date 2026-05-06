import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# ── SQL Engine (in-memory SQLite) ─────────────────────────────────────────────
engine = create_engine("sqlite:///:memory:", echo=False)


def load_transactions_to_sql(df: pd.DataFrame) -> None:
    """Write the transactions DataFrame into SQLite table 'transactions'."""
    df = df.copy()
    # Ensure correct types for SQL
    df["amount"]  = df["amount"].astype(float)
    df["balance"] = pd.to_numeric(df["balance"], errors="coerce")
    df["date"]    = pd.to_datetime(df["date"], errors="coerce")
    df.to_sql("transactions", engine, if_exists="replace", index=False)


def run_sql(query: str) -> pd.DataFrame:
    """Execute a raw SQL query and return results as a DataFrame."""
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


# ── Financial Summary ─────────────────────────────────────────────────────────
def get_financial_summary() -> dict:
    """Core financial metrics via SQL."""
    summary = {}

    queries = {
        "total_income": """
            SELECT COALESCE(SUM(amount), 0) as value
            FROM transactions WHERE amount > 0
        """,
        "total_expenses": """
            SELECT COALESCE(SUM(ABS(amount)), 0) as value
            FROM transactions WHERE amount < 0
        """,
        "net_cashflow": """
            SELECT COALESCE(SUM(amount), 0) as value
            FROM transactions
        """,
        "transaction_count": """
            SELECT COUNT(*) as value FROM transactions
        """,
        "avg_expense": """
            SELECT COALESCE(AVG(ABS(amount)), 0) as value
            FROM transactions WHERE amount < 0
        """,
        "largest_expense": """
            SELECT COALESCE(MAX(ABS(amount)), 0) as value
            FROM transactions WHERE amount < 0
        """,
    }

    for key, query in queries.items():
        result = run_sql(query)
        summary[key] = round(float(result["value"].iloc[0]), 2)

    # Spending by category
    category_df = run_sql("""
        SELECT category,
               SUM(ABS(amount)) as total,
               COUNT(*)         as count
        FROM transactions
        WHERE amount < 0
        GROUP BY category
        ORDER BY total DESC
    """)
    summary["spending_by_category"] = category_df.to_dict(orient="records")

    # Monthly breakdown
    monthly_df = run_sql("""
        SELECT strftime('%Y-%m', date) as month,
               SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END)       as income,
               SUM(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END)  as expenses
        FROM transactions
        GROUP BY month
        ORDER BY month
    """)
    summary["monthly_breakdown"] = monthly_df.to_dict(orient="records")

    return summary


# ── Anomaly Detection ─────────────────────────────────────────────────────────
def detect_anomalies(df: pd.DataFrame, z_threshold: float = 2.5) -> list[dict]:
    """
    Flag anomalous transactions using two methods:
      1. Z-score on expense amounts (per category)
      2. Duplicate detection (same description + amount within 3 days)
    """
    anomalies = []
    expenses = df[df["amount"] < 0].copy()

    if expenses.empty:
        return anomalies

    # -- Method 1: Z-score per category ---
    for category, group in expenses.groupby("category"):
        if len(group) < 3:
            continue
        amounts = group["amount"].abs()
        mean, std = amounts.mean(), amounts.std()
        if std == 0:
            continue
        z_scores = (amounts - mean) / std
        flagged = group[z_scores > z_threshold]
        for _, row in flagged.iterrows():
            anomalies.append({
                "date":        str(row["date"]),
                "description": row["description"],
                "amount":      row["amount"],
                "category":    category,
                "reason":      f"Unusually large for {category} "
                               f"(z={z_scores[row.name]:.2f}, "
                               f"avg=${mean:.2f})",
                "method":      "z_score",
            })

    # -- Method 2: Near-duplicate transactions ---
    expenses["date"] = pd.to_datetime(expenses["date"], errors="coerce")
    expenses_sorted = expenses.sort_values(["description", "date"])
    for _, group in expenses_sorted.groupby("description"):
        if len(group) < 2:
            continue
        group = group.sort_values("date")
        for i in range(1, len(group)):
            prev = group.iloc[i - 1]
            curr = group.iloc[i]
            days_apart = abs((curr["date"] - prev["date"]).days)
            if days_apart <= 3 and curr["amount"] == prev["amount"]:
                anomalies.append({
                    "date":        str(curr["date"]),
                    "description": curr["description"],
                    "amount":      curr["amount"],
                    "category":    curr.get("category", "Other"),
                    "reason":      f"Possible duplicate — same amount "
                                   f"charged {days_apart} day(s) apart",
                    "method":      "duplicate",
                })

    return anomalies


# ── Plotly Charts ─────────────────────────────────────────────────────────────
def plot_spending_by_category(summary: dict) -> go.Figure:
    data = summary.get("spending_by_category", [])
    if not data:
        return go.Figure()
    df = pd.DataFrame(data)
    fig = px.pie(
        df, values="total", names="category",
        title="Spending by Category",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def plot_monthly_cashflow(summary: dict) -> go.Figure:
    data = summary.get("monthly_breakdown", [])
    if not data:
        return go.Figure()
    df = pd.DataFrame(data)
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_trace(go.Bar(
        x=df["month"], y=df["income"],
        name="Income", marker_color="mediumseagreen"
    ))
    fig.add_trace(go.Bar(
        x=df["month"], y=df["expenses"],
        name="Expenses", marker_color="tomato"
    ))
    fig.update_layout(
        title="Monthly Income vs Expenses",
        barmode="group",
        xaxis_title="Month",
        yaxis_title="Amount ($)",
    )
    return fig


def plot_anomalies(df: pd.DataFrame, anomalies: list[dict]) -> go.Figure:
    if df.empty:
        return go.Figure()

    expenses = df[df["amount"] < 0].copy()
    expenses["date"] = pd.to_datetime(expenses["date"], errors="coerce")
    anomaly_dates = {a["description"] for a in anomalies}

    expenses["is_anomaly"] = expenses["description"].isin(anomaly_dates)
    normal   = expenses[~expenses["is_anomaly"]]
    flagged  = expenses[expenses["is_anomaly"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal["date"], y=normal["amount"].abs(),
        mode="markers", name="Normal",
        marker=dict(color="steelblue", size=7, opacity=0.6),
    ))
    fig.add_trace(go.Scatter(
        x=flagged["date"], y=flagged["amount"].abs(),
        mode="markers", name="Anomaly",
        marker=dict(color="red", size=11, symbol="x"),
        text=flagged["description"],
    ))
    fig.update_layout(
        title="Transaction Anomalies",
        xaxis_title="Date",
        yaxis_title="Amount ($)",
    )
    return fig