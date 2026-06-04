"""
Evaluation metrics runner for the Personal Finance Assistant.

Measures:
  - Extraction accuracy  : field-level and transaction-level recall/precision
  - Categorization       : precision/recall/F1 per category + macro averages
  - Anomaly detection    : binary F1 vs golden is_anomaly labels

Two evaluation modes:
  --pipeline   Run the full LLM pipeline (extractor → categorizer → analyzer).
               Requires GROQ_API_KEY to be set.  Saves to baseline_results.json.
  --algorithm  Skip LLM calls; feed golden transactions directly into
               detect_anomalies() to measure the algorithm in isolation.

Usage:
    python eval/metrics_runner.py --pipeline   # full end-to-end eval
    python eval/metrics_runner.py --algorithm  # anomaly algorithm only (no API key needed)
    python eval/metrics_runner.py --pipeline --algorithm  # both
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

GOLDEN_PATH  = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "baseline_results.json"


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_golden() -> list[dict]:
    with open(GOLDEN_PATH) as f:
        data = json.load(f)
    print(f"[golden] Loaded {len(data)} transactions "
          f"({sum(1 for t in data if t['is_anomaly'])} anomalies)")
    return data


def golden_to_statement_text(transactions: list[dict]) -> str:
    """
    Render golden transactions as the same tabular text format that the
    extractor prompt expects (mirrors generate_test_statement.py layout).
    """
    lines = [
        "First National Bank - Statement",
        "Account: REDACTED_ACCOUNT  |  REDACTED_NAME  |  Period: Jan-Mar 2024",
        "",
        f"{'Date':<14}{'Description':<35}{'Amount':>13}{'Balance':>13}",
        "-" * 75,
    ]
    for t in transactions:
        amount_str  = f"${t['amount']:>10,.2f}"
        balance_str = f"${t['balance']:>10,.2f}"
        lines.append(
            f"{t['date']:<14}{t['description']:<35}{amount_str}{balance_str}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Transaction matching
# ─────────────────────────────────────────────────────────────────────────────

def match_transactions(
    extracted: list[dict],
    golden: list[dict],
) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """
    Match extracted transactions to golden ones by (date prefix, amount).
    Returns (matched_pairs, unmatched_extracted, unmatched_golden).
    """
    remaining_gold = list(golden)
    matched: list[tuple[dict, dict]] = []
    unmatched_ext: list[dict] = []

    for ext in extracted:
        ext_date   = str(ext.get("date", ""))[:10]
        ext_amount = round(float(ext.get("amount", 0)), 2)
        found = False
        for i, gold in enumerate(remaining_gold):
            if (
                str(gold["date"])[:10] == ext_date
                and abs(round(float(gold["amount"]), 2) - ext_amount) < 0.02
            ):
                matched.append((ext, gold))
                remaining_gold.pop(i)
                found = True
                break
        if not found:
            unmatched_ext.append(ext)

    return matched, unmatched_ext, remaining_gold


# ─────────────────────────────────────────────────────────────────────────────
# Metric computations
# ─────────────────────────────────────────────────────────────────────────────

def compute_extraction_metrics(
    matched: list[tuple[dict, dict]],
    unmatched_ext: list[dict],
    unmatched_gold: list[dict],
    total_golden: int,
) -> dict:
    tp = len(matched)
    fp = len(unmatched_ext)
    fn = len(unmatched_gold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = _f1(precision, recall)

    # Per-field accuracy on matched pairs
    field_hits = {"date": 0, "amount": 0, "balance": 0, "type": 0}
    for ext, gold in matched:
        if str(ext.get("date", ""))[:10] == str(gold["date"])[:10]:
            field_hits["date"] += 1
        if abs(round(float(ext.get("amount", 0)), 2) - round(float(gold["amount"]), 2)) < 0.02:
            field_hits["amount"] += 1
        ext_bal  = ext.get("balance")
        gold_bal = gold.get("balance")
        if ext_bal is not None and gold_bal is not None:
            if abs(float(ext_bal) - float(gold_bal)) < 0.02:
                field_hits["balance"] += 1
        if str(ext.get("type", "")).lower() == str(gold.get("type", "")).lower():
            field_hits["type"] += 1

    n = len(matched)
    field_accuracy = {k: round(v / n, 4) if n > 0 else 0.0 for k, v in field_hits.items()}

    # Surface misses for debugging
    missed_descriptions = [g["description"] for g in unmatched_gold]
    hallucinated = [
        {"date": e.get("date"), "description": e.get("description"), "amount": e.get("amount")}
        for e in unmatched_ext
    ]

    return {
        "transactions_in_golden":   total_golden,
        "transactions_extracted":   tp + fp,
        "true_positives":           tp,
        "false_positives":          fp,
        "false_negatives":          fn,
        "precision":                round(precision, 4),
        "recall":                   round(recall, 4),
        "f1":                       round(f1, 4),
        "field_accuracy":           field_accuracy,
        "missed_transactions":      missed_descriptions,
        "hallucinated_transactions": hallucinated,
    }


def compute_categorization_metrics(matched: list[tuple[dict, dict]]) -> dict:
    if not matched:
        return {"error": "No matched transactions"}

    y_true = [gold["category"]           for _, gold in matched]
    y_pred = [ext.get("category", "Other") for ext, _  in matched]

    labels = sorted(set(y_true))
    per_category: dict[str, dict] = {}

    for label in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
        support = sum(1 for yt in y_true if yt == label)

        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = _f1(p, r)

        per_category[label] = {
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f1, 4),
            "support":   support,
        }

    correct  = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / len(y_true) if y_true else 0.0

    macro_p  = _mean([v["precision"] for v in per_category.values()])
    macro_r  = _mean([v["recall"]    for v in per_category.values()])
    macro_f1 = _mean([v["f1"]        for v in per_category.values()])

    # Wrong predictions for debugging
    misclassified = [
        {"description": ext.get("description"), "true": yt, "predicted": yp}
        for (ext, gold), yt, yp in zip(matched, y_true, y_pred)
        if yt != yp
    ]

    return {
        "overall_accuracy": round(accuracy, 4),
        "macro_precision":  round(macro_p, 4),
        "macro_recall":     round(macro_r, 4),
        "macro_f1":         round(macro_f1, 4),
        "per_category":     per_category,
        "misclassified":    misclassified,
    }


def compute_anomaly_metrics(
    detected: list[dict],
    golden: list[dict],
    matched: list[tuple[dict, dict]],
) -> dict:
    """
    Binary classification metrics: is each matched transaction correctly
    flagged/unflagged relative to its golden is_anomaly label?
    """
    # Key: (date_str, rounded_amount) → True/False
    detected_keys: set[tuple[str, float]] = set()
    for a in detected:
        date_str = str(a.get("date", ""))[:10]
        detected_keys.add((date_str, round(float(a["amount"]), 2)))

    gold_anomaly_map = {
        (str(g["date"])[:10], round(float(g["amount"]), 2)): g.get("is_anomaly", False)
        for g in golden
    }

    y_true, y_pred = [], []
    for ext, gold in matched:
        key = (str(gold["date"])[:10], round(float(gold["amount"]), 2))
        y_true.append(1 if gold_anomaly_map.get(key, False) else 0)
        ext_key = (str(ext.get("date", ""))[:10], round(float(ext.get("amount", 0)), 2))
        y_pred.append(1 if ext_key in detected_keys else 0)

    if not y_true:
        return {"error": "No matched transactions to evaluate"}

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = _f1(p, r)

    # Surface which anomalies were missed or false-alarmed
    false_negatives = [
        {"description": gold.get("description"), "date": gold.get("date"), "amount": gold.get("amount")}
        for ext, gold in matched
        if gold.get("is_anomaly") and
           (str(ext.get("date", ""))[:10], round(float(ext.get("amount", 0)), 2)) not in detected_keys
    ]
    false_positives_list = [
        {"description": ext.get("description"), "date": ext.get("date"), "amount": ext.get("amount")}
        for ext, gold in matched
        if not gold.get("is_anomaly") and
           (str(ext.get("date", ""))[:10], round(float(ext.get("amount", 0)), 2)) in detected_keys
    ]

    return {
        "golden_anomaly_count": sum(1 for g in golden if g.get("is_anomaly")),
        "detected_count":       len(detected),
        "true_positives":       tp,
        "false_positives":      fp,
        "false_negatives":      fn,
        "true_negatives":       tn,
        "precision":            round(p, 4),
        "recall":               round(r, 4),
        "f1":                   round(f1, 4),
        "missed_anomalies":     false_negatives,
        "false_alarms":         false_positives_list,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm-only baseline (no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def run_algorithm_baseline(golden: list[dict]) -> dict:
    """
    Feed golden transactions directly into detect_anomalies() to measure
    the algorithm's ceiling — what it can detect given perfect extraction.
    """
    from nodes.analyzer import detect_anomalies

    df = pd.DataFrame(golden)
    df["amount"]  = df["amount"].astype(float)
    df["balance"] = pd.to_numeric(df["balance"], errors="coerce")
    df["date"]    = pd.to_datetime(df["date"], errors="coerce")

    detected = detect_anomalies(df)

    # Reuse the same metric logic; supply identity-matched pairs
    matched_pairs = [(g, g) for g in golden]
    return compute_anomaly_metrics(detected, golden, matched_pairs)


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline eval
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline_eval(golden: list[dict]) -> dict:
    """Run golden text through LLM extractor → verifier → categorizer → anomaly detector."""
    from nodes.extractor   import extract_transactions
    from nodes.verifier    import verify_transactions
    from nodes.categorizer import categorize_transactions
    from nodes.analyzer    import detect_anomalies

    statement_text = golden_to_statement_text(golden)

    print("\n[pipeline] Running extractor …")
    ext_result = extract_transactions(statement_text)
    extracted  = ext_result["transactions"]
    print(f"[pipeline] Extracted {len(extracted)} transactions")

    print("[pipeline] Running verifier …")
    ver_result   = verify_transactions(extracted)
    verified     = ver_result["transactions"]
    corrections  = ver_result["corrections_made"]
    schema_fails = ver_result["failed_schema"]
    bal_fails    = ver_result["failed_balance"]
    verified_count = sum(1 for t in verified if t.get("verified"))
    print(f"[pipeline] Verified {verified_count}/{len(verified)} transactions "
          f"({corrections} correction(s) made; "
          f"schema failures: {len(schema_fails)}, balance failures: {len(bal_fails)})")

    print("[pipeline] Running categorizer …")
    cat_result   = categorize_transactions(verified)
    categorized  = cat_result["transactions"]

    print("[pipeline] Running anomaly detector …")
    df = pd.DataFrame(categorized)
    df["amount"]  = df["amount"].astype(float)
    df["balance"] = pd.to_numeric(df.get("balance", pd.Series(dtype=float)), errors="coerce")
    df["date"]    = pd.to_datetime(df["date"], errors="coerce")
    detected = detect_anomalies(df)
    print(f"[pipeline] Detected {len(detected)} anomalies")

    matched, unmatched_ext, unmatched_gold = match_transactions(categorized, golden)
    print(f"[pipeline] Matched {len(matched)}/{len(golden)} transactions")

    return {
        "extraction":    compute_extraction_metrics(matched, unmatched_ext, unmatched_gold, len(golden)),
        "categorization": compute_categorization_metrics(matched),
        "anomaly":       compute_anomaly_metrics(detected, golden, matched),
        "verification":  {
            "total_transactions": len(verified),
            "verified_count":     verified_count,
            "corrections_made":   corrections,
            "schema_failures":    len(schema_fails),
            "balance_failures":   len(bal_fails),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_extraction(m: dict) -> None:
    print("\n── Extraction ──────────────────────────────────────────────")
    print(f"  Golden transactions : {m['transactions_in_golden']}")
    print(f"  Extracted           : {m['transactions_extracted']}")
    print(f"  True positives      : {m['true_positives']}")
    print(f"  False positives     : {m['false_positives']}")
    print(f"  False negatives     : {m['false_negatives']}")
    print(f"  Precision           : {m['precision']:.1%}")
    print(f"  Recall              : {m['recall']:.1%}")
    print(f"  F1                  : {m['f1']:.1%}")
    print("  Field accuracy:")
    for field, acc in m["field_accuracy"].items():
        print(f"    {field:<10}: {acc:.1%}")
    if m["missed_transactions"]:
        print(f"  Missed ({len(m['missed_transactions'])}): {m['missed_transactions'][:5]}")
    if m["hallucinated_transactions"]:
        print(f"  Hallucinated ({len(m['hallucinated_transactions'])}): "
              f"{[h['description'] for h in m['hallucinated_transactions'][:5]]}")


def print_categorization(m: dict) -> None:
    if "error" in m:
        print(f"\n── Categorization ── {m['error']}")
        return
    print("\n── Categorization ──────────────────────────────────────────")
    print(f"  Overall accuracy : {m['overall_accuracy']:.1%}")
    print(f"  Macro precision  : {m['macro_precision']:.1%}")
    print(f"  Macro recall     : {m['macro_recall']:.1%}")
    print(f"  Macro F1         : {m['macro_f1']:.1%}")
    print(f"  {'Category':<22} {'P':>6}  {'R':>6}  {'F1':>6}  {'N':>4}")
    print(f"  {'-'*22} {'------':>6}  {'------':>6}  {'------':>6}  {'----':>4}")
    for cat, v in sorted(m["per_category"].items()):
        print(f"  {cat:<22} {v['precision']:>6.1%}  {v['recall']:>6.1%}  {v['f1']:>6.1%}  {v['support']:>4}")
    if m.get("misclassified"):
        print(f"  Misclassified ({len(m['misclassified'])}):")
        for e in m["misclassified"][:8]:
            print(f"    {e['description']:<30} true={e['true']!r:25} pred={e['predicted']!r}")


def print_anomaly(m: dict, label: str = "Anomaly Detection") -> None:
    if "error" in m:
        print(f"\n── {label} ── {m['error']}")
        return
    print(f"\n── {label} ──────────────────────────────────────────")
    print(f"  Golden anomalies : {m['golden_anomaly_count']}")
    print(f"  Detected         : {m['detected_count']}")
    print(f"  True positives   : {m['true_positives']}")
    print(f"  False positives  : {m['false_positives']}")
    print(f"  False negatives  : {m['false_negatives']}")
    print(f"  True negatives   : {m['true_negatives']}")
    print(f"  Precision        : {m['precision']:.1%}")
    print(f"  Recall           : {m['recall']:.1%}")
    print(f"  F1               : {m['f1']:.1%}")
    if m.get("missed_anomalies"):
        print(f"  Missed: {[a['description'] for a in m['missed_anomalies']]}")
    if m.get("false_alarms"):
        print(f"  False alarms: {[a['description'] for a in m['false_alarms']]}")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline",  action="store_true",
                        help="Run full LLM pipeline eval (requires GROQ_API_KEY)")
    parser.add_argument("--algorithm", action="store_true",
                        help="Run anomaly algorithm baseline on golden data (no API key needed)")
    args = parser.parse_args()

    if not args.pipeline and not args.algorithm:
        parser.print_help()
        sys.exit(0)

    golden = load_golden()
    results: dict = {"generated_at": datetime.utcnow().isoformat() + "Z"}

    if args.algorithm:
        print("\n=== Algorithm Baseline (perfect extraction assumed) ===")
        algo_metrics = run_algorithm_baseline(golden)
        print_anomaly(algo_metrics, label="Anomaly Algorithm Baseline")
        results["algorithm_baseline"] = {"anomaly": algo_metrics}

    if args.pipeline:
        print("\n=== Full Pipeline Eval ===")
        pipeline_metrics = run_pipeline_eval(golden)
        print_extraction(pipeline_metrics["extraction"])
        print_categorization(pipeline_metrics["categorization"])
        print_anomaly(pipeline_metrics["anomaly"])
        v = pipeline_metrics.get("verification", {})
        print("\n── Verification ────────────────────────────────────────────")
        print(f"  Total transactions  : {v.get('total_transactions', 'n/a')}")
        print(f"  Verified (passed)   : {v.get('verified_count', 'n/a')}")
        print(f"  Schema failures     : {v.get('schema_failures', 'n/a')}")
        print(f"  Balance failures    : {v.get('balance_failures', 'n/a')}")
        print(f"  Corrections logged  : {v.get('corrections_made', 'n/a')}")
        results["pipeline"] = pipeline_metrics

    # ── Save ────────────────────────────────────────────────────────────────
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
