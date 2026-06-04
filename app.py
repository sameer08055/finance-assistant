import streamlit as st
import pandas as pd
import uuid
import tempfile
from dotenv import load_dotenv


# ── Correction helpers ────────────────────────────────────────────────────────
def _build_correction_map(verification_report: dict) -> dict:
    """Return {(description, date): correction_detail} from verification_report."""
    corr_map = {}
    for c in verification_report.get("corrections", []):
        key = (c.get("description", ""), str(c.get("date", ""))[:10])
        corr_map[key] = c
    return corr_map


def _humanize_reason(reason: str) -> str:
    r = reason.lower()
    if "balance inconsistency" in r:
        return "Sequential balance inconsistency"
    if "invalid date" in r:
        return "Invalid date format"
    if "invalid amount" in r:
        return "Invalid amount value"
    if "empty" in r and "description" in r:
        return "Missing description"
    if "invalid balance" in r:
        return "Invalid balance value"
    if "invalid type" in r:
        return "Invalid transaction type"
    return reason.split(";")[0].strip().capitalize()

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

    # ── Pipeline Health card (only after a run) ───────────────────────────────
    if st.session_state.pipeline_ran and st.session_state.result:
        vr          = st.session_state.result.get("verification_report", {})
        n_txns      = len(st.session_state.result.get("transactions", []))
        n_verified  = sum(
            1 for t in st.session_state.result.get("transactions", [])
            if t.get("verified")
        )
        n_correct   = vr.get("corrections_made", 0)
        corrections = vr.get("corrections", [])

        st.markdown("---")
        st.markdown("**Pipeline Health**")

        st.markdown(f"✓ Transactions verified: **{n_verified}/{n_txns}**")

        if n_correct == 0:
            st.success("✓ No corrections needed")
        else:
            st.warning(f"⚠ Auto-corrections made: **{n_correct}**")
            last = corrections[-1] if corrections else None
            if last and last.get("fields"):
                f      = last["fields"][0]
                field  = f["field"]
                orig   = f["original_value"]
                fixed  = f["corrected_value"]
                reason = _humanize_reason(last.get("reason", ""))
                st.markdown(
                    f"Last correction:  \n"
                    f"`{field}` &nbsp; `{orig}` → `{fixed}`  \n"
                    f"*{reason}*"
                )

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    if not uploaded_file:
        st.sidebar.error("Please upload a PDF first.")
    else:
        with st.spinner("Processing statement..."):
            # Cloud-safe temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

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
            is_rolling = a.get("method") == "rolling_z_score"

            # ── Prominent summary for rolling z-score (visible without expanding) ─
            if is_rolling and a.get("explanation"):
                st.info(
                    f"**{a['date']} · {a['description']} · "
                    f"${abs(a['amount']):,.2f}**  \n{a['explanation']}"
                )
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Z-Score",        f"{a['z_score']:.2f}σ")
                col_b.metric("90-day Avg",     f"${a['rolling_mean']:,.2f}")
                col_c.metric("90-day Std Dev", f"${a['rolling_std']:,.2f}")

            # ── Detail expander (all anomalies) ──────────────────────────────────
            with st.expander(
                f"⚠️ {a['date']} | {a['description']} | ${abs(a['amount']):,.2f}"
            ):
                st.write(f"**Category:** {a['category']}")
                st.write(f"**Method:**   {a['method'].replace('_', ' ').title()}")
                if a.get("explanation"):
                    st.info(a["explanation"])
                else:
                    st.write(f"**Reason:** {a['reason']}")
                if a.get("z_score") is not None:
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Z-Score",        f"{a['z_score']:.2f}σ")
                    col_b.metric("90-day Avg",     f"${a['rolling_mean']:,.2f}")
                    col_c.metric("90-day Std Dev", f"${a['rolling_std']:,.2f}")

# ── Tab 4: Transactions ───────────────────────────────────────────────────────
with tab4:
    st.subheader("All Transactions")

    if df.empty:
        st.info("No transactions loaded.")
    else:
        # Build correction lookup from verification_report
        vr_report    = result.get("verification_report", {})
        corr_map     = _build_correction_map(vr_report)

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

        # Add ⚠ badge column if any corrections exist
        display_df = filtered.copy()
        if corr_map:
            display_df["Status"] = display_df.apply(
                lambda row: "⚠ Corrected"
                if (row.get("description", ""), str(row.get("date", ""))[:10]) in corr_map
                else "",
                axis=1,
            )

        fmt = {"amount": "${:,.2f}", "balance": "${:,.2f}"}
        st.dataframe(
            display_df.style.format(fmt, na_rep="—"),
            use_container_width=True,
            height=500,
        )
        st.caption(f"Showing {len(filtered)} of {len(df)} transactions")

        # ── Auto-correction detail cards ──────────────────────────────────────
        visible_corrected = [
            corr_map[(row["description"], str(row["date"])[:10])]
            for _, row in filtered.iterrows()
            if (row.get("description", ""), str(row.get("date", ""))[:10]) in corr_map
        ]
        if visible_corrected:
            st.markdown("---")
            st.markdown(f"**Auto-Correction Details** ({len(visible_corrected)} in view)")
            for c in visible_corrected:
                label = (
                    f"⚠ {c['date']} · {c['description']} — "
                    f"{_humanize_reason(c['reason'])}"
                )
                with st.expander(label):
                    st.markdown(
                        f"**Reason:** {_humanize_reason(c['reason'])}"
                    )
                    for f in c.get("fields", []):
                        orig  = f["original_value"]
                        fixed = f["corrected_value"]
                        st.markdown(
                            f"**Field corrected:** `{f['field']}`  \n"
                            f"Original &nbsp; → &nbsp; Corrected  \n"
                            f"`{orig}` &nbsp;→&nbsp; `{fixed}`"
                        )