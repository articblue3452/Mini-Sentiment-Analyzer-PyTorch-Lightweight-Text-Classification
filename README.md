# 📝 Mini Sentiment Analyzer (PyTorch)

This project is a **lightweight sentiment analysis tool** built with **PyTorch**.  
It classifies text into **three categories**:

- ✅ Positive  
- ❌ Negative  
- ➖ Neutral  

The model uses a **Bag-of-Words (BoW)** representation and a **small feed-forward neural network** to train, evaluate, and make predictions.  
All artifacts (trained model, vocabulary, labels) are saved for reuse.  

---

## 🚀 Features
- **3-class sentiment classification** (`positive`, `negative`, `neutral`)
- **Bag-of-Words + Tiny Neural Net** (fast and simple)
- **Training from CSV** with stratified split (train/val)
- **Evaluation loop** (accuracy + loss)
- **Save & load artifacts** (`model.pt`, `vocab.json`, `labels.json`)
- **Prediction** for single text
- **Interactive demo mode**
- **Configurable hyperparameters** via CLI arguments

---

## 📂 Project Structure
```
mini_sentiment.py       # Main script (train, predict, demo)
artifacts/              # Saved model + vocab + labels
sample_data.csv         # Example dataset (text,label)
```

---

## ⚙️ Installation

1. Clone or copy the script.  
2. Install Python dependencies:

```bash
pip install torch
```

*(Optionally also: `numpy`, `pandas`, but not required here.)*

---

## 📊 Dataset Format

Your dataset must be a CSV with **columns:**

```
text,label
I love this movie,positive
This is boring,negative
It's okay,neutral
```

- Labels **must** be one of: `positive`, `negative`, `neutral`.  
- Default dataset path = `sample_data.csv`.

---

## 🏗️ How It Works (Logic Overview)

### 🔹 Tokenization
- Cleans input (removes URLs, mentions, hashtags).  
- Lowercases text.  
- Splits into simple alphanumeric tokens.  

### 🔹 Vocabulary
- Builds a **vocabulary** from training data only.  
- Maps tokens → indices (`stoi`).  
- Encodes each text into a **bag-of-words vector**.  

### 🔹 Dataset & Dataloader
- `SentimentDataset` → wraps text + labels.  
- Each sample: `(vector, label_id)`.  
- Batched with PyTorch `DataLoader`.  

### 🔹 Model (TinySentimentNet)
- **Input:** Bag-of-Words vector.  
- **Layers:**  
  `Linear(input_dim → hidden) → ReLU → Dropout → Linear(hidden → 3 classes)`  
- Outputs **logits** for each sentiment.  

### 🔹 Training
- Optimizer: `Adam`  
- Loss: `CrossEntropyLoss`  
- Stratified train/val split (keeps label distribution balanced).  
- Tracks **val accuracy** and saves best model.  

### 🔹 Prediction
- Loads artifacts (`model.pt`, `vocab.json`, `labels.json`).  
- Encodes new text as BoW.  
- Applies model → probabilities → argmax.  
- Outputs label + confidence + emoji.  

---

## 🖥️ Usage

Run the script as:
```bash
python mini_sentiment.py <command> [options]
```

---

### 🔹 1. Train a model
```bash
python mini_sentiment.py train --data sample_data.csv --epochs 10
```

Options:
- `--data` → CSV dataset (default: `sample_data.csv`)  
- `--epochs` → training epochs (default: 10)  
- `--lr` → learning rate (default: 0.01)  
- `--batch_size` → batch size (default: 32)  
- `--hidden` → hidden layer size (default: 64)  
- `--vocab_size` → max vocab size (default: 10000)  
- `--min_freq` → min token frequency (default: 1)  
- `--cpu` → force CPU (ignore CUDA)  

Example:
```bash
python mini_sentiment.py train --data reviews.csv --epochs 20 --lr 0.005
```

---

### 🔹 2. Predict sentiment for text
```bash
python mini_sentiment.py predict --text "I love MidMax"
```

Example output:
```
Positive ✅ (confidence 0.92)
```

---

### 🔹 3. Interactive demo mode
```bash
python mini_sentiment.py demo
```

Example:
```
Type a sentence (or 'quit'):
> This is the best project
Positive ✅ (confidence 0.87)
> It’s boring
Negative ❌ (confidence 0.91)
> quit
```

---

## 📦 Artifacts Saved

After training, the following files are stored in `artifacts/`:

- `model.pt` → trained model weights  
- `vocab.json` → vocabulary mapping  
- `labels.json` → class labels  

These are used later for prediction.  

---

## 🔮 Future Extensions
- Replace Bag-of-Words with embeddings (Word2Vec, GloVe, BERT).  
- Add more preprocessing (stemming, stopwords).  
- Train with larger datasets (IMDB, Twitter, etc.).  
- Export model to ONNX for deployment.  
