"""Third comparison model: TF-IDF features + logistic regression (classical ML, no pretrained weights).

Run from the project root (venv active):
    python src/tfidf_baseline.py

Same data and conditions as the other models: trains on data/processed/train.csv, picks C on val macro F1,
and is scored on val and test (the independent set is scored by src/evaluate_independent.py, which loads the
saved model). Writes results/tfidf_{val,test}_{report.txt,predictions.csv,confusion.png}, results/tfidf_metrics.json
and models/tfidf_lr.joblib.
"""
import json
import random
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from labels import LABELS

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODEL_PATH = ROOT / "models" / "tfidf_lr.joblib"
SEED = 42
C_GRID = [0.3, 1, 3, 10, 30]
# keep symbols such as + - ( ) = as tokens, because formulas carry much of the signal
TOKEN = r"[A-Za-z]+|\d+|[^\sA-Za-z\d]"


def build(C):
    feats = FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, token_pattern=TOKEN, ngram_range=(1, 2), min_df=2,
                                 sublinear_tf=True)),
        ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                 sublinear_tf=True)),
    ])
    return Pipeline([("tfidf", feats),
                     ("lr", LogisticRegression(C=C, max_iter=3000, random_state=SEED))])


def load(split):
    return pd.read_csv(ROOT / "data" / "processed" / f"{split}.csv")


def evaluate(model, split):
    df = load(split)
    y = df["misconception_label"].to_numpy()
    probs = model.predict_proba(df["response_text"])
    pred = model.classes_[probs.argmax(axis=1)]
    acc = accuracy_score(y, pred)
    mf1 = f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)
    report = classification_report(y, pred, labels=LABELS, zero_division=0)
    print(f"\n=== TF-IDF + logistic regression on {split} ({len(df)} rows) ===")
    print("Accuracy:", round(acc, 3), "| Macro F1:", round(mf1, 3))
    print(report)
    (RESULTS / f"tfidf_{split}_report.txt").write_text(report, encoding="utf-8")
    df.assign(predicted=pred, confidence=probs.max(axis=1).round(3)).to_csv(
        RESULTS / f"tfidf_{split}_predictions.csv", index=False)
    cm = confusion_matrix(y, pred, labels=LABELS)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"TF-IDF + LR confusion matrix ({split})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS / f"tfidf_{split}_confusion.png", dpi=150)
    plt.close()
    return {"accuracy": round(float(acc), 4), "macro_f1": round(float(mf1), 4)}


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    RESULTS.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    train, val = load("train"), load("val")

    best = None
    for C in C_GRID:
        m = build(C).fit(train["response_text"], train["misconception_label"])
        pred = m.predict(val["response_text"])
        f1 = f1_score(val["misconception_label"], pred, labels=LABELS, average="macro", zero_division=0)
        print(f"C={C:<5} validation macro F1 = {f1:.3f}")
        if best is None or f1 > best[0]:
            best = (f1, C, m)
    print(f"Chosen C = {best[1]} (selected on validation only)")
    model = best[2]
    joblib.dump(model, MODEL_PATH)

    metrics = {"C": best[1], "val": evaluate(model, "val"), "test": evaluate(model, "test")}
    (RESULTS / "tfidf_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nSaved", MODEL_PATH)


if __name__ == "__main__":
    main()
