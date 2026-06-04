# Personal Finance Assistant

A local, privacy-first personal finance assistant built with LangChain, LangGraph, and Streamlit. Upload your bank statement PDF and get instant financial analysis, spending insights, anomaly detection, and an AI-powered chat interface — all without your data leaving your machine.

---

## Features

- **PDF Ingestion** — Upload any bank statement PDF via drag and drop
- **PII Redaction** — Automatically redacts account numbers, SSNs, phone numbers, emails, and names using Regex + Presidio before any LLM sees your data
- **Transaction Extraction** — LLM-powered extraction of dates, descriptions, amounts, and balances
- **Self-Correcting Pipeline** — Extracted transactions are validated for schema, type, and balance consistency; failures are automatically re-extracted by the LLM and logged to `eval/corrections.log`
- **Smart Categorization** — Automatically categorizes transactions into 16 spending categories
- **SQL Analytics** — Accurate financial calculations (income, expenses, cashflow, averages) via SQLite
- **Semantic Search** — FAISS + HuggingFace embeddings for natural language transaction search
- **Anomaly Detection** — Flags unusual charges via rolling 90-day z-score per category and duplicate detection (same description + amount within 3 days)
- **Interactive Charts** — Plotly pie charts and bar charts for spending and monthly cashflow
- **Conversational Memory** — Ask follow-up questions with full context via LangGraph MemorySaver
- **Eval Framework** — Measures extraction accuracy, categorization F1, and anomaly detection performance against a golden dataset
- **LangSmith Tracing** — Full pipeline observability out of the box

---

## Architecture

```
PDF Upload
│
▼
PII Redaction (Regex + Presidio)
│
▼
Transaction Extraction (Groq LLM)
│
▼
Verification + Self-Correction (schema, type, balance checks → LLM re-extraction)
│
▼
Categorization (Groq LLM)
│
▼
SQL Analysis + Anomaly Detection (SQLite + rolling z-score)
│
▼
FAISS Vectorstore (HuggingFace Embeddings)
│
▼
LLM Recommender (Groq LLM + Memory)
│
▼
Streamlit UI
```

---

## Project Structure

```
finance_assistant/
├── app.py                      # Streamlit UI
├── graph.py                    # LangGraph state, nodes, edges
├── nodes/
│   ├── pii.py                  # PII redaction pipeline
│   ├── extractor.py            # Transaction extraction
│   ├── verifier.py             # Schema validation + self-correcting re-extraction
│   ├── categorizer.py          # Transaction categorization
│   ├── analyzer.py             # SQL analytics, rolling z-score anomaly detection, Plotly charts
│   └── recommender.py          # LLM financial advisor
├── eval/
│   ├── golden_dataset.json     # Labeled ground-truth transactions
│   ├── metrics_runner.py       # Extraction, categorization, and anomaly eval runner
│   ├── edge_case_test.py       # Edge case test suite
│   ├── baseline_results.json   # Saved eval results
│   └── corrections.log         # Log of LLM self-corrections
├── utils/
│   ├── pdf_loader.py           # PDF text extraction
│   └── vectorstore.py          # FAISS build + semantic search
├── generate_test_statement.py  # Generate a sample PDF for testing
└── requirements.txt
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/sameer08055/finance-assistant.git
cd finance-assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=finance-assistant
```

Get your keys:
- Groq: [console.groq.com](https://console.groq.com)
- LangSmith: [smith.langchain.com](https://smith.langchain.com)

### 4. Run the app

```bash
streamlit run app.py
```

---

## Testing with a Sample Statement

Generate a sample bank statement PDF with built-in PII and a duplicate transaction for anomaly testing:

```bash
pip install fpdf2
python generate_test_statement.py
```

Then upload `test_statement.pdf` via the Streamlit sidebar.

---

## Evaluation

Run the eval framework against the golden dataset to measure pipeline quality:

```bash
# Full end-to-end eval (requires GROQ_API_KEY)
python eval/metrics_runner.py --pipeline

# Anomaly algorithm only (no API key needed)
python eval/metrics_runner.py --algorithm

# Both
python eval/metrics_runner.py --pipeline --algorithm
```

Metrics reported: extraction field-level recall/precision, categorization precision/recall/F1 per category (+ macro averages), and anomaly detection binary F1.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq `llama-3.1-8b-instant` |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Orchestration | LangGraph |
| Memory | LangGraph MemorySaver |
| Vector Store | FAISS |
| PDF Parsing | pdfplumber |
| PII Redaction | Presidio + Regex |
| Database | SQLite via SQLAlchemy |
| Visualization | Plotly |
| UI | Streamlit |
| Tracing | LangSmith |

---

## Privacy

All processing is local. Your bank statement is:
1. Never stored to disk beyond the session
2. PII-redacted before being sent to any LLM API
3. Cleared when you start a new session

The only external API calls are to Groq (redacted text only) and LangSmith (traces).

---

## License

MIT
