#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mini Sentiment Analyzer (PyTorch)
- 3 classes: positive / negative / neutral
- Bag-of-Words features + small neural net
- Train, evaluate, save artifacts, predict

Usage:
  python mini_sentiment.py train --data sample_data.csv --epochs 10
  python mini_sentiment.py predict --text "I love MidMax"
  python mini_sentiment.py demo
"""
import os
import re
import csv
import json
import math
import time
import random
import argparse
from collections import Counter, defaultdict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ART_DIR = "artifacts"
DEFAULT_DATA = "sample_data.csv"
LABELS = ["negative", "neutral", "positive"]  # fixed order

# ----------------------
# Utils
# ----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def simple_tokenize(text: str):
    """Very simple preprocessing + tokenization."""
    text = text.lower()
    # remove urls, mentions, hashtags
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    # keep words (alphanumeric + apostrophes)
    tokens = re.findall(r"[a-z0-9']+", text)
    return tokens

def read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            txt = (r.get("text") or "").strip()
            lab = (r.get("label") or "").strip().lower()
            if txt and lab in {"positive","negative","neutral"}:
                rows.append((txt, lab))
    return rows

def stratified_split(rows, val_ratio=0.15, seed=42):
    by_label = defaultdict(list)
    for txt, lab in rows:
        by_label[lab].append((txt, lab))
    random.Random(seed).shuffle(rows)
    for k in by_label:
        random.Random(seed).shuffle(by_label[k])
    train, val = [], []
    for lab, items in by_label.items():
        n = len(items)
        n_val = max(1, int(n * val_ratio))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    random.Random(seed).shuffle(train)
    random.Random(seed).shuffle(val)
    return train, val

# ----------------------
# Vocabulary & Vectorizer
# ----------------------
class Vocab:
    def __init__(self, tokens, max_size=10000, min_freq=1):
        cnt = Counter(tokens)
        # reserve 0 for <unk>
        self.itos = ["<unk>"]
        for w, c in cnt.most_common():
            if c < min_freq:
                continue
            if w not in self.itos:
                self.itos.append(w)
            if len(self.itos) >= max_size:
                break
        self.stoi = {w:i for i,w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode_bow(self, tokens):
        vec = torch.zeros(len(self), dtype=torch.float32)
        for t in tokens:
            idx = self.stoi.get(t, 0)  # <unk>
            vec[idx] += 1.0
        return vec

# ----------------------
# Dataset
# ----------------------
LABEL_TO_ID = {"negative":0, "neutral":1, "positive":2}

class SentimentDataset(Dataset):
    def __init__(self, rows, vocab: Vocab):
        self.rows = rows
        self.vocab = vocab

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        text, label = self.rows[idx]
        tokens = simple_tokenize(text)
        x = self.vocab.encode_bow(tokens)  # [V]
        y = LABEL_TO_ID[label]
        return x, y

def collate(batch):
    xs, ys = zip(*batch)
    X = torch.stack(xs, dim=0)  # [B, V]
    y = torch.tensor(ys, dtype=torch.long)
    return X, y

# ----------------------
# Model
# ----------------------
class TinySentimentNet(nn.Module):
    def __init__(self, input_dim, hidden=64, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ----------------------
# Training & Eval
# ----------------------
def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            loss_sum += loss.item() * X.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += X.size(0)
    return correct / max(1,total), loss_sum / max(1,total)

def train(args):
    set_seed(args.seed)

    data_path = args.data if args.data is not None else DEFAULT_DATA
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    rows = read_csv(data_path)
    if not rows:
        raise RuntimeError("No valid rows found in CSV (need columns text,label).")

    # Build vocab from training texts only
    train_rows, val_rows = stratified_split(rows, val_ratio=0.15, seed=args.seed)
    vocab_tokens = []
    for txt, _ in train_rows:
        vocab_tokens.extend(simple_tokenize(txt))
    vocab = Vocab(vocab_tokens, max_size=args.vocab_size, min_freq=args.min_freq)

    # Datasets + loaders
    train_ds = SentimentDataset(train_rows, vocab)
    val_ds   = SentimentDataset(val_rows, vocab)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_ld   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = TinySentimentNet(input_dim=len(vocab), hidden=args.hidden).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Vocab size: {len(vocab)} | Train: {len(train_ds)} | Val: {len(val_ds)} | Device: {device}")

    best_val_acc = 0.0
    os.makedirs(ART_DIR, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        for X, y in train_ld:
            X, y = X.to(device), y.to(device)
            optim.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * X.size(0)
            seen += X.size(0)

        train_loss = epoch_loss / max(1, seen)
        val_acc, val_loss = evaluate(model, val_ld, device)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}%")

        # Save best
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(ART_DIR, "model.pt"))
            with open(os.path.join(ART_DIR, "vocab.json"), "w", encoding="utf-8") as f:
                json.dump({"itos": vocab.itos}, f, ensure_ascii=False, indent=2)
            with open(os.path.join(ART_DIR, "labels.json"), "w", encoding="utf-8") as f:
                json.dump({"labels": LABELS}, f, ensure_ascii=False, indent=2)
            print(f"  ↳ Saved artifacts (best so far: {best_val_acc*100:.2f}%)")

    print("Training complete. Best val acc: {:.2f}%".format(best_val_acc*100))

def load_artifacts():
    # Load vocab & labels
    with open(os.path.join(ART_DIR, "vocab.json"), "r", encoding="utf-8") as f:
        itos = json.load(f)["itos"]
    vocab = Vocab(tokens=[], max_size=1)  # dummy init
    vocab.itos = itos
    vocab.stoi = {w:i for i,w in enumerate(itos)}

    with open(os.path.join(ART_DIR, "labels.json"), "r", encoding="utf-8") as f:
        labels = json.load(f)["labels"]
    model = TinySentimentNet(input_dim=len(vocab), hidden=64)
    model.load_state_dict(torch.load(os.path.join(ART_DIR, "model.pt"), map_location="cpu"))
    model.eval()
    return model, vocab, labels

def predict(text: str):
    model, vocab, labels = load_artifacts()
    tokens = simple_tokenize(text)
    x = vocab.encode_bow(tokens).unsqueeze(0)  # [1, V]
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        idx = int(torch.argmax(probs).item())
        label = labels[idx]
        conf = float(probs[idx].item())
    emoji = {"positive":"✅", "negative":"❌", "neutral":"➖"}[label]
    print(f"{label.capitalize()} {emoji} (confidence {conf:.2f})")

def demo():
    print("Type a sentence (or 'quit'):")
    while True:
        try:
            s = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not s or s.lower() in {"quit","exit"}:
            break
        predict(s)

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--data", type=str, default=None, help="CSV with columns text,label")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--lr", type=float, default=1e-2)
    p_train.add_argument("--batch_size", type=int, default=32)
    p_train.add_argument("--hidden", type=int, default=64)
    p_train.add_argument("--vocab_size", type=int, default=10000)
    p_train.add_argument("--min_freq", type=int, default=1)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--cpu", action="store_true", help="Force CPU")

    p_pred = sub.add_parser("predict", help="Predict sentiment for a single text")
    p_pred.add_argument("--text", type=str, required=True)

    sub.add_parser("demo", help="Interactive prediction loop")

    args = parser.parse_args()

    if args.cmd == "train":
        if args.data is None:
            args.data = DEFAULT_DATA
        train(args)
    elif args.cmd == "predict":
        predict(args.text)
    elif args.cmd == "demo":
        demo()

if __name__ == "__main__":
    main()
