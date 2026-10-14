import re
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from labels import LABELS

ROOT = Path(__file__).resolve().parent.parent

STRONG = r"HCl|HBr|HI|HNO3|H2SO4|HClO4"
WEAK = r"HF|CH3COOH|HCN|H2CO3|HNO2|H3PO4|HCOOH|HClO"

# (label, pattern, optional exclusion pattern). First match wins, so order matters.
RULES = [
    # --- independence vs mutually exclusive ---
    ("independence_confusion",
     r"mutually exclusive.{0,60}\bindependent\b|\bindependent\b.{0,60}(mutually exclusive|disjoint)|disjoint.{0,60}\bindependent\b",
     r"\b(not|isn't|aren't|different|differ)\b"),
    ("independence_confusion", r"\bindependent\b.{0,40}\b(cannot|can't|cant|never)\b.{0,30}(happen|occur|overlap)", None),
    ("independence_confusion", r"\bindependent\b.{0,30}P\(.{1,20}\)\s*=\s*0\b", None),
    ("independence_confusion", r"\bindependent\b.{0,50}(empty intersection|prevents|exclude each other|do not overlap|never overlap)", None),
    ("independence_confusion", r"(can|could) (both )?(happen|occur) together.{0,40}\bdependent\b", None),
    ("independence_confusion", r"P\(.{1,20}\)\s*=\s*0 then .{1,20} (are|is) independent", None),

    # --- probability rules ---
    ("probability_rule_misuse",
     r"P\((\w+) or (\w+)\)\s*=\s*P\(\1\)\s*\+\s*P\(\2\)(?!\s*-)",
     r"only when|mutually exclusive|disjoint"),
    ("probability_rule_misuse", r"P\((\w+) and (\w+)\)\s*=\s*P\(\1\)\s*\+\s*P\(\2\)", None),
    ("probability_rule_misuse", r"and B\)\s*(is )?always P\(A\) times", None),
    ("probability_rule_misuse", r"P\((\w+)\s*(given|\|)\s*(\w+)\)\s*(=|equals|is the same as)\s*P\(\3\s*(given|\|)\s*\1\)", None),
    ("probability_rule_misuse", r"P\(not A\)\s*=\s*(1/P\(A\)|P\(A\)\s*-\s*1)", None),
    ("probability_rule_misuse", r"P\(A or B\).{0,40}(multiply|times)", None),
    ("probability_rule_misuse", r"P\(\w+\)\s*=\s*1\.\d", None),
    ("probability_rule_misuse", r"\d\.\d\s*\+\s*\d\.\d\s*=\s*1\.\d", None),
    ("probability_rule_misuse", r"P\(A\|B\)\s*=\s*P\(A and B\)\s*/\s*P\(A\)\s*$", None),
    ("probability_rule_misuse", r"without replacement.{0,60}original probabilities", None),
    ("probability_rule_misuse", r"P\([^)]*\)\s*=\s*(1\.(?!0+\b)\d+|[2-9]\.\d+)", None),

    # --- function notation ---
    ("function_notation", r"\b([fgh])\(([^()]*\+[^()]*)\)\s*(=|means|is the same as|is)\s*\1\(", None),
    ("function_notation", r"\b([fgh])\(\w+\)\s*(means|is)\s*\1\s*(times|multiplied)", None),
    ("function_notation", r"parenthes\w+.{0,30}mean\s+multipl", None),
    ("function_notation", r"\b([fgh])\(\d*x\)\s*(is|=)\s*(just )?\d+\s*\1\(x\)", None),
    ("function_notation", r"\^-1\(x\).{0,30}1\s*/", None),
    ("function_notation", r"tells you to multiply", None),
    ("function_notation", r"\b([fgh])\(x\)\s*=\s*x\^2\s*(so|then)\s*\1\(x\s*\+\s*\d+\)\s*=\s*x\^2\s*\+\s*\d+\s*$", None),
    ("function_notation", r"\b([fgh])\(x\)\s*=\s*(\d+)x\s*(so|then)\s*\1\(x\s*\+\s*\d+\)\s*=\s*\2x\s*\+\s*(?!\2\b)\d+\s*$", None),
    ("function_notation", r"\b([fgh])\(\d*x\)\s*(is|=)\s*(just )?\d+\s*(times|\*)\s*\1\(x\)", r"linear|only|\bif\b"),
    ("function_notation", r"\b([fgh])\(x\)\s*=.+\b\1\(\d+\)\s*=\s*[^=]*x", None),
    ("function_notation", r"\b([fgh])\(\1\(x\)\)\s*(is|=)\s*(just )?\1\(x\)\s*(squared|\^2)", None),
    ("function_notation", r"\b([fgh])\(x\) and \1 are the same", None),

    # --- conjugate pairs ---
    ("conjugate_confusion", r"conjugate base of \S+( \d-)? (is|=) (H\+|H3O\+|\S+\+)", None),
    ("conjugate_confusion", r"(H\+|\S+\+) is the conjugate base", None),
    ("conjugate_confusion", r"conjugate base.{0,40}(gains|gain|accepts|picks up) a proton", None),
    ("conjugate_confusion", r"conjugate.{0,40}electron", None),
    ("conjugate_confusion", r"\bH\+ is the conjugate (acid|base)", None),
    ("conjugate_confusion", r"conjugate (acid|base) of \S+(?: \d-)? (is|=) H\+", None),
    ("conjugate_confusion", r"you get H\+ as the conjugate", None),
    ("conjugate_confusion", r"you get \S+\+ as the conjugate base", None),
    ("conjugate_confusion", r"conjugate base of (?!H2O\b)\S+(?: \d-)? (is|=) OH-", None),
    ("conjugate_confusion", r"\bOH- is the conjugate base of (?!H2O\b)", None),
    ("conjugate_confusion", r"when (?!H2O\b)\S+ donates a proton you get OH- as the conjugate base", None),

    # --- strong vs weak ---
    ("strong_weak_confusion", rf"\b({STRONG})\b.{{0,60}}\bweak\b", r"conjugate"),
    ("strong_weak_confusion", rf"\b({WEAK})\b.{{0,60}}\bstrong\b", r"\bnot\b|conjugate"),
    ("strong_weak_confusion", r"\b(strong|weak)\b.{0,40}\b(concentrat\w+|dilute)\b|\b(concentrat\w+|dilute)\b.{0,40}\b(strong|weak)\b", None),
    ("strong_weak_confusion", r"strong.{0,30}(high|low) pH|weak acid.{0,20}pH of 7", None),
    ("strong_weak_confusion", r"weak acids? (do not|dont|don't) dissociate", None),
]

COMPILED = [(lab, re.compile(p, re.I), re.compile(x, re.I) if x else None)
            for lab, p, x in RULES]


def predict(text):
    for label, pat, excl in COMPILED:
        if pat.search(text) and not (excl and excl.search(text)):
            return label
    return "correct"


def evaluate(split):
    df = pd.read_csv(ROOT / "data" / "processed" / f"{split}.csv")
    y_true = df["misconception_label"]
    y_pred = df["response_text"].map(predict)

    report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)
    print(f"\n=== Baseline on {split} ({len(df)} rows) ===")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 3))
    print(report)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"baseline_{split}_report.txt").write_text(report)
    df.assign(predicted=y_pred).to_csv(out / f"baseline_{split}_predictions.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"Baseline confusion matrix ({split})")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(out / f"baseline_{split}_confusion.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    for split in ("val", "test"):
        evaluate(split)