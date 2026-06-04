"""
Edge-case test: rolling-window z-score vs. static z-score.

Scenario
--------
A user spent heavily on Food & Dining for the first three months
($80-$120/meal, 12 transactions).  Over the next two months they
switched to cheap eats ($8-$14/meal, 10 transactions).
Then a single $75 restaurant charge appears.

Static z-score uses ALL history → the $75 charge looks ordinary
  (it sits near the historical mean of ~$60).
Rolling 90-day z-score uses ONLY the recent window → the $75 charge
  is far above the recent mean of ~$11 and gets flagged.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from nodes.analyzer import detect_anomalies   # rolling implementation


# ── Build synthetic dataset ───────────────────────────────────────────────────
# Key constraint: Phase 1 must be >90 days before the test transaction so that
# the rolling window sees ONLY Phase 2 cheap-eats data.

rng = np.random.default_rng(42)

anchor = pd.Timestamp("2024-03-20")   # "today" for the test

# Phase 1: 200–123 days before anchor  (heavy spender, 12 transactions)
phase1_dates  = [anchor - pd.Timedelta(days=200 - i * 7) for i in range(12)]
phase1_amounts = rng.uniform(80, 120, 12).round(2)

# Phase 2: 80–17 days before anchor  (cheap eats, 10 transactions)
phase2_dates  = [anchor - pd.Timedelta(days=80 - i * 7) for i in range(10)]
phase2_amounts = rng.uniform(8, 14, 10).round(2)

# The test transaction — on the anchor date itself
test_date   = anchor
test_amount = 75.00

all_dates   = list(phase1_dates)   + list(phase2_dates)   + [test_date]
all_amounts = list(phase1_amounts) + list(phase2_amounts) + [test_amount]

df = pd.DataFrame({
    "date":        all_dates,
    "description": ["RESTAURANT"] * len(all_dates),
    "amount":      [-a for a in all_amounts],   # negative = expense
    "balance":     [10000.0] * len(all_dates),
    "type":        ["debit"] * len(all_dates),
    "category":    ["Food & Dining"] * len(all_dates),
})
df["date"] = pd.to_datetime(df["date"])

# ── Print dataset summary ─────────────────────────────────────────────────────
all_abs = [abs(a) for a in df["amount"]]
ph1_abs = all_abs[:12]
ph2_abs = all_abs[12:22]

print("=" * 65)
print("EDGE CASE: rolling window vs. static z-score")
print("=" * 65)
print(f"\nPhase 1 (heavy spender, 12 txns): "
      f"mean=${np.mean(ph1_abs):.2f}  std=${np.std(ph1_abs, ddof=1):.2f}")
print(f"Phase 2 (cheap eats,  10 txns): "
      f"mean=${np.mean(ph2_abs):.2f}  std=${np.std(ph2_abs, ddof=1):.2f}")
print(f"\nTest transaction: ${test_amount:.2f} on {test_date.date()}")

# ── Static z-score (reference implementation) ────────────────────────────────
prior_amounts = np.array(all_abs[:-1])   # everything except the test txn
static_mean   = prior_amounts.mean()
static_std    = prior_amounts.std(ddof=1)
static_z      = (test_amount - static_mean) / static_std if static_std else 0

print("\n── Static z-score (all history) ──────────────────────────")
print(f"  Prior mean : ${static_mean:.2f}")
print(f"  Prior std  : ${static_std:.2f}")
print(f"  z-score    : {static_z:.2f}")
print(f"  Flagged?   : {'YES ⚠️' if static_z > 2.5 else 'NO  ✓'}")

# ── Rolling window z-score (new implementation) ──────────────────────────────
rolling_detected = detect_anomalies(df.copy(), z_threshold=2.5, window_days=90, min_window=5)
z_flags = [a for a in rolling_detected if a["method"] == "rolling_z_score"]

print("\n── Rolling 90-day z-score (new implementation) ───────────")
if z_flags:
    a = z_flags[0]
    print(f"  Window mean : ${a['rolling_mean']:.2f}")
    print(f"  Window std  : ${a['rolling_std']:.2f}")
    print(f"  z-score     : {a['z_score']:.2f}")
    print(f"  Flagged?    : YES ⚠️")
    print(f"  Explanation : {a['explanation']}")
else:
    print("  Flagged?    : NO  ✓")

# ── Verdict ───────────────────────────────────────────────────────────────────
static_missed  = static_z <= 2.5
rolling_caught = len(z_flags) > 0

print("\n── Verdict ───────────────────────────────────────────────")
if static_missed and rolling_caught:
    print("  PASS: static MISSED it, rolling CAUGHT it.")
    print("  The rolling window detects a recent spending pattern")
    print("  change that all-time history would wash out.")
elif not static_missed and rolling_caught:
    print("  Both methods flagged it (static threshold also triggered).")
elif not rolling_caught:
    print("  FAIL: rolling window did not flag the transaction.")
print("=" * 65)
