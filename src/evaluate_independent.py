"""Score all three models (frozen keyword baseline, TF-IDF + logistic regression, fine-tuned DistilBERT) on an
independent test set. Run src/tfidf_baseline.py and src/train.py first so both saved models exist.

Run from the project root (venv active):
    python src/evaluate_independent.py
    python src/evaluate_independent.py path\\to\\other_file.csv

Input CSV (default data/raw/independent_test.csv), one response per row:
    response_text,misconception_label,writer
The writer column is optional. Labels must be the six names in src/labels.py.

IMPORTANT: this is the final exam. Run it once the file is complete, report whatever it prints, and do not edit
rules, templates or training data in response to these rows. If you change anything, say so in the paper and
collect a fresh set.

Writes to results/: independent_{baseline,tfidf,model}_report.txt, _predictions.csv, _confusion.png,
independent_summary.csv, independent_high_conf_errors.csv
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from labels import LABELS                      # noqa: E402
import baseline                                # noqa: E402  (frozen v2 rules; predict(text) -> label)
from app.main import (SUPERSCRIPTS, classifier, looks_in_scope, prepare)  # noqa: E402

TFIDF_PATH = ROOT / "models" / "tfidf_lr.joblib"
DEFAULT_CSV = ROOT / "data" / "raw" / "independent_test.csv"
RESULTS = ROOT / "results"
HIGH_CONF = 0.80
BOOT = 2000


def clean(text):
    return re.sub(r"\s+", " ", str(text).translate(SUPERSCRIPTS)).strip()


def simple_norm(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def load(path):
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        # Excel's plain "CSV" option saves in the Windows code page, not UTF-8
        df = pd.read_csv(path, encoding="cp1252")
    missing = {"response_text", "misconception_label"} - set(df.columns)
    if missing:
        sys.exit(f"{path} is missing columns: {sorted(missing)}")
    df = df.dropna(subset=["response_text", "misconception_label"]).copy()
    df["response_text"] = df["response_text"].map(clean)
    df["misconception_label"] = df["misconception_label"].astype(str).str.strip()
    df = df[df["response_text"] != ""]
    bad = sorted(set(df["misconception_label"]) - set(LABELS))
    if bad:
        sys.exit(f"Unknown labels in {path}: {bad}\nAllowed: {LABELS}")
    return df.reset_index(drop=True)


def overlap_report(df):
    seen = set()
    for split in ("train", "val", "test"):
        f = ROOT / "data" / "processed" / f"{split}.csv"
        if f.exists():
            seen |= {simple_norm(t) for t in pd.read_csv(f)["response_text"]}
    dup = [t for t in df["response_text"] if simple_norm(t) in seen]
    print(f"Rows that already appear (after normalizing) in train/val/test: {len(dup)}")
    for t in dup[:10]:
        print("   ", t)
    return len(dup)


def bootstrap_ci(y_true, y_pred, rng):
    n = len(y_true)
    accs, f1s = [], []
    for _ in range(BOOT):
        idx = rng.integers(0, n, n)
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1s.append(f1_score(y_true[idx], y_pred[idx], labels=LABELS, average="macro", zero_division=0))
    lo = lambda a: round(float(np.percentile(a, 2.5)), 3)
    hi = lambda a: round(float(np.percentile(a, 97.5)), 3)
    return (lo(accs), hi(accs)), (lo(f1s), hi(f1s))


def save_outputs(name, df, pred):
    y_true = df["misconception_label"].to_numpy()
    report = classification_report(y_true, pred, labels=LABELS, zero_division=0)
    (RESULTS / f"independent_{name}_report.txt").write_text(report, encoding="utf-8")
    df.assign(predicted=pred).to_csv(RESULTS / f"independent_{name}_predictions.csv", index=False)
    cm = confusion_matrix(y_true, pred, labels=LABELS)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{name} confusion matrix (independent test set)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS / f"independent_{name}_confusion.png", dpi=150)
    plt.close()
    return report


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    if not path.exists():
        sys.exit(f"Not found: {path}\nCreate it from independent_test_template.csv first.")
    RESULTS.mkdir(exist_ok=True)
    df = load(path)
    y_true = df["misconception_label"].to_numpy()

    print(f"Loaded {len(df)} rows from {path}")
    counts = df["misconception_label"].value_counts().reindex(LABELS, fill_value=0)
    print(counts.to_string())
    if counts.min() < 5:
        print("WARNING: a label has fewer than 5 rows, so its per-class scores will be very noisy.")
    if "writer" in df.columns:
        print("Rows per writer:", df["writer"].value_counts().to_dict())
    overlap_report(df)

    # baseline (same cleaned text the baseline was evaluated on before)
    base_pred = df["response_text"].map(baseline.predict).to_numpy()

    # TF-IDF + logistic regression (trained by src/tfidf_baseline.py)
    if not TFIDF_PATH.exists():
        sys.exit(f"Missing {TFIDF_PATH}. Run: python src/tfidf_baseline.py")
    tfidf = joblib.load(TFIDF_PATH)
    tfidf_pred = tfidf.predict(df["response_text"])

    # model (same prompt format and max length as training; confidence = top softmax score)
    classifier.load()
    model_pred, conf = [], []
    for t in df["response_text"]:
        scores = classifier.scores(prepare(t))
        label, p = max(scores.items(), key=lambda kv: kv[1])
        model_pred.append(label)
        conf.append(round(p, 3))
    model_pred = np.array(model_pred)
    conf = np.array(conf)

    base_report = save_outputs("baseline", df, base_pred)
    tfidf_report = save_outputs("tfidf", df, tfidf_pred)
    mdf = df.assign(confidence=conf, guard_would_reject=[not looks_in_scope(t) for t in df["response_text"]])
    model_report = save_outputs("model", mdf, model_pred)

    rng = np.random.default_rng(42)
    rows = []
    for name, pred, report in (("Baseline v2", base_pred, base_report), ("TF-IDF + LR", tfidf_pred, tfidf_report),
                               ("DistilBERT", model_pred, model_report)):
        acc = accuracy_score(y_true, pred)
        mf1 = f1_score(y_true, pred, labels=LABELS, average="macro", zero_division=0)
        (a_lo, a_hi), (f_lo, f_hi) = bootstrap_ci(y_true, pred, rng)
        print(f"\n=== {name} on the independent set ({len(df)} rows) ===")
        print(f"Accuracy {acc:.3f} (95% bootstrap CI {a_lo}-{a_hi}) | Macro F1 {mf1:.3f} (CI {f_lo}-{f_hi})")
        print(report)
        rows.append((name, acc, mf1, pred))

    per_class = pd.DataFrame({
        "label": LABELS,
        "support": counts.values,
        "baseline_f1": f1_score(y_true, base_pred, labels=LABELS, average=None, zero_division=0).round(2),
        "tfidf_f1": f1_score(y_true, tfidf_pred, labels=LABELS, average=None, zero_division=0).round(2),
        "model_f1": f1_score(y_true, model_pred, labels=LABELS, average=None, zero_division=0).round(2),
    })
    summary = pd.concat([per_class, pd.DataFrame([
        {"label": "MACRO F1", "support": len(df), "baseline_f1": round(rows[0][2], 3), "tfidf_f1": round(rows[1][2], 3), "model_f1": round(rows[2][2], 3)},
        {"label": "ACCURACY", "support": len(df), "baseline_f1": round(rows[0][1], 3), "tfidf_f1": round(rows[1][1], 3), "model_f1": round(rows[2][1], 3)},
    ])])
    summary.to_csv(RESULTS / "independent_summary.csv", index=False)
    print("\nPer-class F1 (baseline, TF-IDF + LR, DistilBERT):\n" + summary.to_string(index=False))

    wrong = mdf.assign(predicted=model_pred)
    wrong = wrong[(wrong["predicted"] != wrong["misconception_label"]) & (wrong["confidence"] >= HIGH_CONF)]
    wrong.to_csv(RESULTS / "independent_high_conf_errors.csv", index=False)
    n_err = int((model_pred != y_true).sum())
    print(f"\nModel errors: {n_err} of {len(df)}; {len(wrong)} of them at confidence >= {HIGH_CONF} "
          f"(see results/independent_high_conf_errors.csv)")
    fp = int(((model_pred != "correct") & (y_true == "correct")).sum())
    fn = int(((model_pred == "correct") & (y_true != "correct")).sum())
    print(f"Correct answers flagged as misconceptions: {fp}; misconceptions passed as correct: {fn}")
    if mdf["guard_would_reject"].any():
        print(f"NOTE: the app's out-of-scope check would reject {int(mdf['guard_would_reject'].sum())} of these rows.")


if __name__ == "__main__":
    main()
