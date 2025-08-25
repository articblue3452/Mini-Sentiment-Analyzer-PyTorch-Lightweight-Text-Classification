# Mini Sentiment Analyzer (PyTorch) — Positive / Negative / Neutral

A tiny, fast sentiment classifier you can train in minutes on CPU. It uses:
- Simple **bag‑of‑words** features (counts of words).
- A small **PyTorch** neural net (Linear → ReLU → Linear) to classify into 3 labels: `positive`, `negative`, `neutral`.

> Perfect for learning the full pipeline: data → preprocessing → features → model → training → metrics → saving → inference.

---

## Quickstart

```bash
# 1) Create a virtual env (optional but recommended)
python -m venv .venv && .venv\Scripts\activate    # on Windows
# source .venv/bin/activate                          # on macOS/Linux

# 2) Install deps
pip install torch pandas

# 3) Train (uses included sample_data.csv by default)
python mini_sentiment.py train --epochs 10 --lr 0.01

# 4) Predict interactively
python mini_sentiment.py demo

# 5) One-off prediction
python mini_sentiment.py predict --text "I love MidMax"
```

If you have your **own CSV dataset**, use:
```bash
python mini_sentiment.py train --data my_data.csv --epochs 8
```
CSV format should be: `text,label` with label in `{{positive,negative,neutral}}`.

---

## Files
- `mini_sentiment.py` — All-in-one script (training + prediction).
- `sample_data.csv` — Small tri-class dataset to get started.
- `artifacts/` — Saved model + vocabulary + label map after training.

---

## How it works (Logic Overview)

1. **Preprocess text**: lowercasing, remove URLs/mentions/hashtags, keep alphanumeric tokens (very simple tokenizer).
2. **Build vocabulary** from training texts (top-N most frequent words, default 10k). We include a special `<unk>` token for unseen words.
3. **Vectorize** each text into a **bag-of-words count vector** of shape `[vocab_size]`.
4. **Model** (PyTorch):
   - `Linear(vocab_size → 64)` → `ReLU` → `Dropout(0.2)` → `Linear(64 → 3)`.
   - Trained with `CrossEntropyLoss` and `Adam`.
5. **Train/Val split**: 85% train / 15% validation (stratified-ish via per-class shuffling).
6. **Metrics**: accuracy on validation set printed each epoch.
7. **Save artifacts**: `artifacts/model.pt`, `artifacts/vocab.json`, `artifacts/labels.json`.
8. **Inference**: load artifacts, vectorize new text, run the model, output label + score.

This is a teaching-friendly baseline. You can later swap features (e.g., TF‑IDF) or the model (e.g., a tiny Transformer).

---

## Tips
- For better generalization, train on more labeled data.
- Add simple **emoji/negation handling** to squeeze a bit more accuracy.
- Switch to **TF‑IDF** features for a stronger linear baseline.
- For deep learning fun, replace the BOW encoder with an **Embedding + mean-pool** or a **tiny Transformer**.
