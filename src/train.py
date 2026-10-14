"""Fine-tune DistilBERT to classify student responses into misconception labels.

Run from the project root:   python src/train.py

Reads : data/processed/train.csv, val.csv, test.csv
Saves : models/distilbert-misconception/            (best epoch by validation macro F1)
        results/model_{val,test}_report.txt          (precision / recall / F1 per class)
        results/model_{val,test}_confusion.png       (confusion matrices)
        results/model_{val,test}_predictions.csv     (text, true label, predicted label, confidence)
        results/model_metrics.json                   (headline numbers)

Uses a plain PyTorch training loop (no Trainer) so it works across transformers versions.
The first run downloads distilbert-base-uncased (about 270 MB), so it needs an internet connection.
"""
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from labels import LABELS

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "distilbert-base-uncased"
SAVE_DIR = ROOT / "models" / "distilbert-misconception"
RESULTS = ROOT / "results"

MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
SEED = 42

label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fmt(text):
    """Prompt-style formatting (part of the preprocessing plan)."""
    return f"Student response: {text}"


def load_split(split):
    df = pd.read_csv(ROOT / "data" / "processed" / f"{split}.csv")
    texts = df["response_text"].map(fmt).tolist()
    labels = df["misconception_label"].map(label2id).tolist()
    return df, texts, labels


def make_loader(texts, labels, tokenizer, shuffle):
    items = list(zip(texts, labels))

    def collate(batch):
        t, y = zip(*batch)
        enc = tokenizer(list(t), padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt")
        enc["labels"] = torch.tensor(y)
        return enc

    return DataLoader(items, batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        out.append(torch.softmax(model(**batch).logits, dim=-1).cpu().numpy())
    return np.concatenate(out)


def evaluate_split(model, tokenizer, split):
    df, texts, labels = load_split(split)
    probs = predict_probs(model, make_loader(texts, labels, tokenizer, shuffle=False))
    pred = probs.argmax(axis=1)
    y_true = np.array(labels)

    acc = accuracy_score(y_true, pred)
    macro_f1 = f1_score(y_true, pred, average="macro")
    report = classification_report(y_true, pred, labels=list(range(len(LABELS))),
                                   target_names=LABELS, zero_division=0)
    print(f"\n=== DistilBERT on {split} ({len(df)} rows) ===")
    print("Accuracy:", round(acc, 3), "| Macro F1:", round(macro_f1, 3))
    print(report)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"model_{split}_report.txt").write_text(report, encoding="utf-8")
    out = df.copy()
    out["predicted"] = [id2label[i] for i in pred]
    out["confidence"] = probs.max(axis=1).round(3)
    out.to_csv(RESULTS / f"model_{split}_predictions.csv", index=False)

    cm = confusion_matrix(y_true, pred, labels=list(range(len(LABELS))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"DistilBERT confusion matrix ({split})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS / f"model_{split}_confusion.png", dpi=150)
    plt.close()
    return {"accuracy": round(float(acc), 4), "macro_f1": round(float(macro_f1), 4)}


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=id2label, label2id=label2id).to(device)

    _, tr_x, tr_y = load_split("train")
    _, va_x, va_y = load_split("val")
    train_loader = make_loader(tr_x, tr_y, tokenizer, shuffle=True)
    val_loader = make_loader(va_x, va_y, tokenizer, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    best_f1, start = -1.0, time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % 25 == 0:
                print(f"epoch {epoch}/{EPOCHS} step {step}/{len(train_loader)} "
                      f"loss {running / step:.4f} ({time.time() - start:.0f}s elapsed)")

        pred = predict_probs(model, val_loader).argmax(axis=1)
        f1 = f1_score(va_y, pred, average="macro")
        acc = accuracy_score(va_y, pred)
        print(f"--> epoch {epoch}: val accuracy {acc:.3f}, val macro F1 {f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"    saved best model to {SAVE_DIR}")

    print(f"\nTraining finished in {(time.time() - start) / 60:.1f} minutes. Best val macro F1: {best_f1:.3f}")

    # evaluate the saved best model on both splits
    model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR).to(device)
    tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
    metrics = {"val": evaluate_split(model, tokenizer, "val"),
               "test": evaluate_split(model, tokenizer, "test"),
               "settings": {"model": MODEL_NAME, "max_len": MAX_LEN, "batch_size": BATCH_SIZE,
                            "epochs": EPOCHS, "lr": LR, "seed": SEED}}
    (RESULTS / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nSaved reports, confusion matrices, and predictions to", RESULTS)


if __name__ == "__main__":
    main()
