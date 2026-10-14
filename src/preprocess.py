"""Preprocess v2: group-aware split for the expanded dataset.

Run from the project root:   python src/preprocess.py
Input : data/synthetic/expanded.csv      (from expand_dataset.py)
        data/raw/handwritten_test.csv    (your held-out test set)
Output: data/processed/train.csv, val.csv, test.csv
"""
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from expand_dataset import norm

ROOT = Path(__file__).resolve().parent.parent
SUPERSCRIPTS = str.maketrans("⁺⁻₀₁₂₃₄₅₆₇₈₉", "+-0123456789")


def clean(text):
    text = str(text).translate(SUPERSCRIPTS)
    return re.sub(r"\s+", " ", text).strip()


syn = pd.read_csv(ROOT / "data" / "synthetic" / "expanded.csv")
test = pd.read_csv(ROOT / "data" / "raw" / "handwritten_test.csv")

for df in (syn, test):
    df["response_text"] = df["response_text"].map(clean)

syn["key"] = syn["response_text"].map(norm)

# 1) the same text must not appear under two different labels
labels_per_key = syn.groupby("key")["misconception_label"].nunique()
conflicts = labels_per_key[labels_per_key > 1].index
syn = syn[~syn["key"].isin(conflicts)]
print(f"Dropped {len(conflicts)} texts that had conflicting labels")

# 2) deduplicate
before = len(syn)
syn = syn.drop_duplicates(subset="key")
print(f"Dropped {before - len(syn)} duplicate rows")

# 3) leakage check against the handwritten test set
test_keys = set(test["response_text"].map(norm))
before = len(syn)
syn = syn[~syn["key"].isin(test_keys)]
print(f"Removed {before - len(syn)} rows that overlapped the test set")

syn = syn.reset_index(drop=True)

# 4) split by template group (about 1/7 of the groups go to validation, stratified by label)
sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
train_idx, val_idx = next(sgkf.split(syn, syn["misconception_label"], syn["group_id"]))
train = syn.iloc[train_idx].drop(columns="key")
val = syn.iloc[val_idx].drop(columns="key")
assert not set(train["group_id"]) & set(val["group_id"]), "group leakage between train and val"

out = ROOT / "data" / "processed"
out.mkdir(parents=True, exist_ok=True)
train.to_csv(out / "train.csv", index=False)
val.to_csv(out / "val.csv", index=False)
test.to_csv(out / "test.csv", index=False)

print("train/val/test sizes:", len(train), len(val), len(test))
print("\nTrain label counts:\n", train["misconception_label"].value_counts())
print("\nVal label counts:\n", val["misconception_label"].value_counts())