import streamlit as st
import pandas as pd
import uuid
from dotenv import load_dotenv

from utils.pdf_loader import load_pdf_pages, combine_pages
from graph import run_pipeline, run_followup
from nodes.analyzer import (
    plot_spending_by_category,
    plot_monthly_cashflow,
    plot_anomalies,
)

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Personal Finance Assistant",
    page_icon="💰",
    layout="wide",
)

# ── Session state defaults ────────────────────────────────────────────────────
if "thread_id"    not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "pipeline_ran" not in st.session_state:
    st.session_state.pipeline_ran = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "result"       not in st.session_state:
    st.session_state.result = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💰 Finance Assistant")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload Bank Statement (PDF)",
        type=["pdf"],
    )

    first_question = st.text_input(
        "Initial question",
        value="Give me a summary of my finances.",
    )

    run_btn = st.button("Analyse Statement", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption(f"Session ID: `{st.session_state.thread_id[:8]}...`")

    if st.button("🔄 New Session", use_container_width=True):
        for key in ["thread_id", "pipeline_ran", "chat_history", "result"]:
            del st.session_state[key]
        st.rerun()

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    if not uploaded_file:
        st.sidebar.error("Please upload a PDF first.")
    else:
        with st.spinner("Processing statement..."):
            tmp_path = f"/tmp/{uploaded_file.name}"
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.read())

            pages    = load_pdf_pages(tmp_path)
            raw_text = combine_pages(pages)

            result = run_pipeline(
                raw_text=  raw_text,
                question=  first_question,
                thread_id= st.session_state.thread_id,
            )

        if result.get("error"):
            st.error(f"Pipeline error: {result['error']}")
        else:
            st.session_state.result       = result
            st.session_state.pipeline_ran = True
            st.session_state.chat_history = [
                {"role": "user",      "content": first_question},
                {"role": "assistant", "content": result["answer"]},
            ]
            st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
if not st.session_state.pipeline_ran:
    st.title("Personal Finance Assistant")
    st.info("Upload a bank statement PDF in the sidebar to get started.")
    st.stop()

result    = st.session_state.result
summary   = result.get("summary", {})
anomalies = result.get("anomalies", [])
df        = pd.DataFrame(result.get("transactions", []))

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Chat", "📊 Dashboard", "⚠️ Anomalies", "📋 Transactions"
])

# ── Tab 1: Chat ───────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Chat with your finances")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a follow-up question..."):
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = run_followup(
                    question=  prompt,
                    thread_id= st.session_state.thread_id,
                )
            st.markdown(answer)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )

# ── Tab 2: Dashboard ──────────────────────────────────────────────────────────
with tab2:
    st.subheader("Financial Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Income",   f"${summary.get('total_income', 0):,.2f}")
    col2.metric("Total Expenses", f"${summary.get('total_expenses', 0):,.2f}")
    col3.metric("Net Cashflow",   f"${summary.get('net_cashflow', 0):,.2f}")
    col4.metric("Transactions",   summary.get("transaction_count", 0))

    st.markdown("---")

    col_left, col_right = st.columns(2)
    with col_left:
        fig_pie = plot_spending_by_category(summary)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_right:
        fig_bar = plot_monthly_cashflow(summary)
        st.plotly_chart(fig_bar, use_container_width=True)

# ── Tab 3: Anomalies ──────────────────────────────────────────────────────────
with tab3:
    st.subheader("Anomaly Detection")

    if not anomalies:
        st.success("No anomalies detected in your statement.")
    else:
        st.warning(f"{len(anomalies)} anomalies detected.")

        fig_anomaly = plot_anomalies(df, anomalies)
        st.plotly_chart(fig_anomaly, use_container_width=True)

        for a in anomalies:
            with st.expander(
                f"⚠️ {a['date']} | {a['description']} | ${abs(a['amount']):,.2f}"
            ):
                st.write(f"**Category:** {a['category']}")
                st.write(f"**Reason:**   {a['reason']}")
                st.write(f"**Method:**   {a['method'].replace('_', ' ').title()}")

# ── Tab 4: Transactions ───────────────────────────────────────────────────────
with tab4:
    st.subheader("All Transactions")

    if df.empty:
        st.info("No transactions loaded.")
    else:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            categories = ["All"] + sorted(df["category"].unique().tolist())
            cat_filter = st.selectbox("Category", categories)
        with col_f2:
            type_filter = st.selectbox("Type", ["All", "debit", "credit"])
        with col_f3:
            search_term = st.text_input("Search description")

        filtered = df.copy()
        if cat_filter  != "All":
            filtered = filtered[filtered["category"] == cat_filter]
        if type_filter != "All":
            filtered = filtered[filtered["type"] == type_filter]
        if search_term:
            filtered = filtered[
                filtered["description"].str.contains(search_term, case=False, na=False)
            ]

        st.dataframe(
            filtered.style.format({
                "amount":  "${:,.2f}",
                "balance": "${:,.2f}",
            }),
            use_container_width=True,
            height=500,
        )
        st.caption(f"Showing {len(filtered)} of {len(df)} transactions")