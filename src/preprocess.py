import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent

SUPERSCRIPTS = str.maketrans("⁺⁻₀₁₂₃₄₅₆₇₈₉", "+-0123456789")
STARTERS = ["i think ", "so ", "basically ", "my answer is that ",
            "well ", "i remember that ", "in my notes it says "]

def clean(text):
    text = str(text).translate(SUPERSCRIPTS)
    return re.sub(r"\s+", " ", text).strip()

def norm(text):
    t = text.lower()
    for s in STARTERS:
        if t.startswith(s):
            t = t[len(s):]
    return t

syn = pd.read_csv(ROOT / "data" / "synthetic" / "synthetic_raw.csv")
test = pd.read_csv(ROOT / "data" / "raw" / "handwritten_test.csv")

for df in (syn, test):
    df["response_text"] = df["response_text"].map(clean)

# dedupe synthetic data on normalized text
syn["key"] = syn["response_text"].map(norm)
syn = syn.drop_duplicates(subset="key")

# leakage check: drop synthetic rows that match any test row
test_keys = set(test["response_text"].map(norm))
before = len(syn)
syn = syn[~syn["key"].isin(test_keys)].drop(columns="key")
print(f"Removed {before - len(syn)} rows that overlapped the test set")

train, val = train_test_split(
    syn, test_size=0.15, stratify=syn["misconception_label"], random_state=42)

out = ROOT / "data" / "processed"
out.mkdir(parents=True, exist_ok=True)
train.to_csv(out / "train.csv", index=False)
val.to_csv(out / "val.csv", index=False)
test.to_csv(out / "test.csv", index=False)

print("train/val/test sizes:", len(train), len(val), len(test))
print(train["misconception_label"].value_counts())