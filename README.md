# BiGRU-Attention: Hybrid Sentiment Classification and Generative Text Modeling

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io)
[![Dataset](https://img.shields.io/badge/Dataset-IMDB%2050k-yellow?style=flat-square)](https://ai.stanford.edu/~amaas/data/sentiment/)
[![Colab](https://img.shields.io/badge/Trained%20on-Google%20Colab%20T4-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![FAST NUCES](https://img.shields.io/badge/Institution-FAST%20NUCES-blue?style=flat-square)](https://nu.edu.pk)

> **Research Paper:** *Enhancing Sentiment Classification and Generative Text Modeling Through Hybrid BiGRU with Multi-Head Self-Attention: An Extension of Deep Learning Model Analysis*
> **Abyaz Israr** (22i-2056) — FAST NUCES, Islamabad

---

## Overview

This project extends the comparative deep learning analysis of Shiri et al. (2023) in two directions:

1. **BiGRU-Attention** — a novel hybrid architecture combining stacked Bidirectional GRU layers with 4-head Multi-Head Self-Attention, residual connections, and layer normalization for sentiment classification
2. **Character-Level LSTM Generator** — a generative language model producing novel movie review text with temperature-controlled sampling — directly addressing the generative AI gap absent from the original study

---

## Results at a Glance

### Classification — IMDB Test Set (N = 25,000)

| Model | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| RNN | 85.44% | 85.47% | 85.33% | 85.61% |
| LSTM | 86.14% | 86.16% | 86.05% | 86.28% |
| BiLSTM | 86.42% | 86.17% | 87.76% | 84.64% |
| GRU | 84.04% | 83.84% | 84.92% | 82.78% |
| BiGRU | 85.58% | 85.58% | 85.60% | 85.55% |
| **BiGRU-Attention (ours)** | **87.65%** | **87.21%** | **87.93%** | **86.50%** |

**+2.07% accuracy over BiGRU baseline · +1.23% over strongest baseline (BiLSTM)**
**622 fewer false positives · 851 fewer false negatives vs BiGRU baseline**

### Generative — Character-Level LSTM

| Temperature | Behaviour | Sample |
|---|---|---|
| T = 0.4 | Fluent, repetitive | *"the film was great and the story was very well done..."* |
| T = 0.7 | Natural, balanced | *"the story was quite interesting though some parts felt a bit slow..."* |
| T = 1.0 | Creative, noisy | *"the plome sartid with a gread stary and the coreacters were intreasting..."* |

Generator training loss: **2.90 → 2.27** over 40 epochs (21.7% reduction)

---

## Architecture

### BiGRU-Attention (Proposed)

```
Input Sequence (150 tokens)
        │
        ▼
Embedding Layer (64-dim, trainable)
        │
        ▼
BiGRU Layer 1 (96 units/direction → 192-dim, return_sequences=True)
Dropout(0.1) + Recurrent Dropout(0.1)
        │
        ▼
BiGRU Layer 2 (48 units/direction → 96-dim, return_sequences=True)
        │
        ▼
Multi-Head Self-Attention (4 heads, key_dim=48)
        │
        ▼
Residual Connection + Layer Normalization
        │
        ▼
Global Average Pooling → Dense(64, ReLU) → Dropout(0.4) → Dense(1, Sigmoid)
        │
        ▼
Binary Sentiment Label (Positive / Negative)
```

### Character-Level LSTM Generator

```
Seed Text (80 characters)
        │
        ▼
LSTM(128, return_sequences=True) → Dropout(0.3)
LSTM(128) → Dropout(0.3)
Dense(|V|=38, Softmax)
        │
        ▼
Temperature-Scaled Sampling (T ∈ {0.4, 0.7, 1.0})
        │
        ▼
Generated Text (250 characters)
```

---

## Repository Structure

```
bigru-attention-nlp/
│
├── notebooks/
│   ├── 01_classification_baselines.ipynb   ← RNN, LSTM, BiLSTM, GRU, BiGRU
│   ├── 02_bigru_attention.ipynb            ← proposed BiGRU-Attention model
│   └── 03_character_lstm_generator.ipynb   ← generative model + temperature sampling
│
├── src/
│   ├── models.py                           ← all model definitions
│   ├── preprocessing.py                    ← tokenization, padding, corpus prep
│   ├── train.py                            ← training with EarlyStopping
│   ├── evaluate.py                         ← accuracy, F1, confusion matrix
│   └── generate.py                         ← text generation with temperature control
│
├── results/
│   ├── classification_results.csv          ← all model metrics
│   ├── confusion_matrices/                 ← saved plots
│   ├── training_curves/                    ← accuracy and loss curves
│   └── generated_samples.txt              ← generated text at T=0.4, 0.7, 1.0
│
├── docs/
│   └── Research_Paper.pdf                  ← full paper
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Clone and install
```bash
git clone https://github.com/Abyazisra/bigru-attention-nlp.git
cd bigru-attention-nlp
pip install -r requirements.txt
```

### 2. Run in Google Colab (recommended)
Open any notebook in `notebooks/` — IMDB loads automatically:
```python
from tensorflow.keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
```

### 3. Train the proposed model
```bash
python src/train.py --model bigru_attention
```

### 4. Evaluate all models
```bash
python src/evaluate.py
```

### 5. Generate text
```bash
python src/generate.py --temperature 0.7 --length 250
```

---

## Key Implementation Details

**Classification preprocessing**
- Vocabulary: top 10,000 words
- Max sequence length: 150 tokens
- Padding: pre-sequence (preserves sentiment-critical end of review)

**Generative preprocessing**
- Corpus: first 4,000 training reviews (~150,000 characters)
- Window: 80 characters, stride 4 → ~37,500 training examples

**Training**
- Optimizer: Adam · Batch size: 256
- EarlyStopping: patience=5, restore best weights
- Environment: Google Colab (Tesla T4 GPU)
- Random seed: 42

---

## Citation

```bibtex
@article{israr2025bigru,
  title={Enhancing Sentiment Classification and Generative Text Modeling
         Through Hybrid BiGRU with Multi-Head Self-Attention},
  author={Israr, Abyaz},
  institution={FAST NUCES, Islamabad, Pakistan},
  year={2025}
}
```

---

## References

- [Shiri et al., 2023](https://arxiv.org/abs/2305.17473) — base comparative study this work extends
- [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) — Attention is All You Need
- [Bahdanau et al., 2015](https://arxiv.org/abs/1409.0473) — Neural MT with Attention
- [Maas et al., 2011](https://ai.stanford.edu/~amaas/data/sentiment/) — IMDB Dataset
- [Karpathy, 2015](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) — Unreasonable Effectiveness of RNNs

---

*Department of Artificial Intelligence and Data Science · FAST NUCES, Islamabad*
