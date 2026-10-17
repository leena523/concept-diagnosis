"""Paired comparison of the three models on the independent test set (same 47 rows, same order).

Run after src/evaluate_independent.py, from the project root:
    python src/compare_models.py
Reads results/independent_{baseline,tfidf,model}_predictions.csv. Prints, for each pair of models, how many rows
one got right and the other wrong, a 95% paired-bootstrap interval for the accuracy difference, and an exact
two-sided sign-test p-value on the disagreements. With a sample this small, treat these as rough guides.
"""
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
NAMES = {"baseline": "Baseline v2", "tfidf": "TF-IDF + LR", "model": "DistilBERT"}
BOOT = 10000


def sign_test_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    frames = {k: pd.read_csv(RESULTS / f"independent_{k}_predictions.csv") for k in NAMES}
    base = frames["baseline"]
    for k, f in frames.items():
        if not (f["response_text"].tolist() == base["response_text"].tolist()):
            raise SystemExit(f"Rows of {k} are not in the same order as the baseline file.")
    truth = base["misconception_label"].to_numpy()
    correct = {k: (f["predicted"].to_numpy() == truth).astype(int) for k, f in frames.items()}
    n = len(truth)

    print(f"Rows: {n}")
    for k in NAMES:
        print(f"  {NAMES[k]:<12} {correct[k].sum()} correct ({correct[k].mean():.3f})")

    rng = np.random.default_rng(42)
    idx = rng.integers(0, n, size=(BOOT, n))
    print("\nPaired comparisons (accuracy of first minus second):")
    for a, b in (("tfidf", "model"), ("model", "baseline"), ("tfidf", "baseline")):
        d = correct[a] - correct[b]
        boot = d[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        only_a = int(((correct[a] == 1) & (correct[b] == 0)).sum())
        only_b = int(((correct[a] == 0) & (correct[b] == 1)).sum())
        print(f"  {NAMES[a]} - {NAMES[b]}: diff {d.mean():+.3f} (95% CI {lo:+.3f} to {hi:+.3f}); "
              f"{NAMES[a]} only right {only_a}, {NAMES[b]} only right {only_b}; sign-test p = {sign_test_p(only_a, only_b):.3f}")


if __name__ == "__main__":
    main()
