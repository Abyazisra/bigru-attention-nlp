# =============================================================================
# NLP Comparative Analysis + Generative Text Generation  
# Based on: "A Comprehensive Overview and Comparative Analysis on Deep Learning
# Models: CNN, RNN, LSTM, GRU" - Shiri et al., arXiv:2305.17473 (2023)

# Author: Abyaz Israr 22i-2056
# =============================================================================

import os, re, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Suppress TF info/warning logs — keeps terminal clean
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Embedding, SimpleRNN, LSTM, GRU,
    Bidirectional, Dense, Dropout, GlobalAveragePooling1D,
    MultiHeadAttention, LayerNormalization, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

# =============================================================================
# 0.  REPRODUCIBILITY
# =============================================================================
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# =============================================================================
# 1.  HYPER-PARAMETERS  
# =============================================================================
VOCAB_SIZE   = 10_000
MAX_LEN      = 150      
EMBED_DIM    = 64       
BATCH_SIZE   = 256      
EPOCHS       = 15       
DROPOUT_RATE = 0.4

RNN_LR       = 3e-4
DEFAULT_LR   = 1e-3
HIDDEN_BASE  = 64
HIDDEN_BI    = 64
HIDDEN_ATT   = 96       

GEN_SEQ_LEN    = 80     
GEN_STEP       = 4      
GEN_EPOCHS     = 40     
GEN_BATCH      = 256    
GEN_LSTM_UNITS = 128    
GEN_TEMPS      = [0.4, 0.7, 1.0]
GEN_LENGTH     = 250

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 65)
print("  NLP Comparative Analysis + Generative Text  [v3 FAST]")
print("  Estimated runtime: 30-45 minutes on CPU")
print("=" * 65)

# =============================================================================
# 2.  LOAD IMDB DATA
# =============================================================================
print("\n[1/8] Loading IMDB dataset ...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
print(f"      Train: {len(X_train)}  |  Test: {len(X_test)}")

X_train = pad_sequences(X_train, maxlen=MAX_LEN, padding='pre', truncating='pre')
X_test  = pad_sequences(X_test,  maxlen=MAX_LEN, padding='pre', truncating='pre')
print(f"      Shape: {X_train.shape}")

# =============================================================================
# 3.  MODEL DEFINITIONS
# =============================================================================
print("\n[2/8] Defining models ...")

def build_rnn(name="RNN"):
    inp = Input(shape=(MAX_LEN,))
    x   = Embedding(VOCAB_SIZE, EMBED_DIM)(inp)
    x   = SimpleRNN(HIDDEN_BASE)(x)
    x   = Dropout(DROPOUT_RATE)(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model(inp, out, name=name)

def build_lstm(name="LSTM"):
    inp = Input(shape=(MAX_LEN,))
    x   = Embedding(VOCAB_SIZE, EMBED_DIM)(inp)
    x   = LSTM(HIDDEN_BASE)(x)
    x   = Dropout(DROPOUT_RATE)(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model(inp, out, name=name)

def build_bilstm(name="BiLSTM"):
    inp = Input(shape=(MAX_LEN,))
    x   = Embedding(VOCAB_SIZE, EMBED_DIM)(inp)
    x   = Bidirectional(LSTM(HIDDEN_BI))(x)
    x   = Dropout(DROPOUT_RATE)(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model(inp, out, name=name)

def build_gru(name="GRU"):
    inp = Input(shape=(MAX_LEN,))
    x   = Embedding(VOCAB_SIZE, EMBED_DIM)(inp)
    x   = GRU(HIDDEN_BASE, recurrent_dropout=0.1)(x)
    x   = Dropout(DROPOUT_RATE)(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model(inp, out, name=name)

def build_bigru(name="BiGRU"):
    inp = Input(shape=(MAX_LEN,))
    x   = Embedding(VOCAB_SIZE, EMBED_DIM)(inp)
    x   = Bidirectional(GRU(HIDDEN_BI))(x)
    x   = Dropout(DROPOUT_RATE)(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model(inp, out, name=name)

def build_bigru_attention(name="BiGRU_Attention"):
    """
    PROPOSED IMPROVEMENT:
    Embed -> BiGRU(96, return_seq) -> BiGRU(48, return_seq)
    -> MultiHead-Attention(4 heads) -> Residual+LayerNorm
    -> GlobalAvgPool -> Dense(64) -> Output
    """
    inp = Input(shape=(MAX_LEN,))
    x   = Embedding(VOCAB_SIZE, EMBED_DIM)(inp)

    # Stacked BiGRU layers
    x   = Bidirectional(GRU(HIDDEN_ATT, return_sequences=True,
                            dropout=0.1, recurrent_dropout=0.1))(x)
    x   = Dropout(DROPOUT_RATE)(x)
    x   = Bidirectional(GRU(HIDDEN_ATT // 2, return_sequences=True,
                            dropout=0.1))(x)
    x   = Dropout(DROPOUT_RATE)(x)

    # 4-head self-attention
    attn = MultiHeadAttention(num_heads=4,
                              key_dim=HIDDEN_ATT // 2,
                              dropout=0.1)(x, x)
    attn = Dropout(DROPOUT_RATE)(attn)

    # Residual + LayerNorm
    x   = Add()([x, attn])
    x   = LayerNormalization(epsilon=1e-6)(x)

    # Classification head
    x   = GlobalAveragePooling1D()(x)
    x   = Dense(64, activation='relu')(x)
    x   = Dropout(DROPOUT_RATE)(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model(inp, out, name=name)

MODEL_CONFIG = [
    ("RNN",             build_rnn,             RNN_LR,     True),
    ("LSTM",            build_lstm,            DEFAULT_LR, False),
    ("BiLSTM",          build_bilstm,          DEFAULT_LR, False),
    ("GRU",             build_gru,             RNN_LR,     True),
    ("BiGRU",           build_bigru,           DEFAULT_LR, False),
    ("BiGRU_Attention", build_bigru_attention, DEFAULT_LR, False),
]

# =============================================================================
# 4.  TRAINING LOOP
# =============================================================================
print("\n[3/8] Training classification models ...\n")

histories   = {}
predictions = {}
metrics     = {}

for idx, (mname, builder, lr, use_clip) in enumerate(MODEL_CONFIG):
    model = builder(name=mname)
    print(f"  [{idx+1}/6] Training {mname} ...")

    opt = Adam(learning_rate=lr, clipnorm=1.0) if use_clip \
          else Adam(learning_rate=lr)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    cbs = [
        EarlyStopping(monitor='val_loss', patience=5,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, verbose=0)
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cbs,
        verbose=1          # show epoch progress so you know it is running
    )

    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)

    histories[mname]   = history.history
    predictions[mname] = y_pred
    metrics[mname] = {
        "Accuracy"  : round(acc  * 100, 2),
        "F1-Score"  : round(f1   * 100, 2),
        "Precision" : round(prec * 100, 2),
        "Recall"    : round(rec  * 100, 2),
    }
    epochs_ran = len(history.history['loss'])
    print(f"  --> {mname}: Acc={acc*100:.2f}%  F1={f1*100:.2f}%  "
          f"Ran {epochs_ran} epochs\n")
    tf.keras.backend.clear_session()

# =============================================================================
# 5.  RESULTS TABLE
# =============================================================================
print("\n[4/8] Results Summary\n")
print(f"{'Model':<22} {'Accuracy':>10} {'F1-Score':>10} "
      f"{'Precision':>11} {'Recall':>8}")
print("-" * 65)
for mname, m in metrics.items():
    print(f"{mname:<22} {m['Accuracy']:>9}%  {m['F1-Score']:>9}%"
          f"  {m['Precision']:>9}%  {m['Recall']:>7}%")

# =============================================================================
# 6.  PLOTS
# =============================================================================
print("\n[5/8] Generating plots ...")
COLORS = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#a65628']

# ── Accuracy curves ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for i, (mname, hist) in enumerate(histories.items()):
    ax = axes.flatten()[i]
    ax.plot(hist['accuracy'],     color=COLORS[i], lw=2, label='Train')
    ax.plot(hist['val_accuracy'], color=COLORS[i], lw=2, ls='--', label='Val')
    ax.set_title(mname, fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.legend(fontsize=8); ax.set_ylim([0.4, 1.0]); ax.grid(alpha=0.3)
plt.suptitle('Training vs Validation Accuracy — All Models',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_accuracy_curves.png"),
            dpi=150, bbox_inches='tight')
plt.close(); print("     Saved: 01_accuracy_curves.png")

# Loss curves 
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for i, (mname, hist) in enumerate(histories.items()):
    ax = axes.flatten()[i]
    ax.plot(hist['loss'],     color=COLORS[i], lw=2, label='Train')
    ax.plot(hist['val_loss'], color=COLORS[i], lw=2, ls='--', label='Val')
    ax.set_title(mname, fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle('Training vs Validation Loss — All Models',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_loss_curves.png"),
            dpi=150, bbox_inches='tight')
plt.close(); print("     Saved: 02_loss_curves.png")

# Accuracy bar chart 
model_names = list(metrics.keys())
accuracies  = [metrics[m]['Accuracy'] for m in model_names]
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(model_names, accuracies, color=COLORS,
              edgecolor='black', width=0.6)
for bar, val in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f"{val}%", ha='center', fontsize=11, fontweight='bold')
y_min = max(0, min(accuracies) - 8)
ax.set_ylim([y_min, 100])
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Model Accuracy Comparison on IMDB Dataset',
             fontsize=14, fontweight='bold')
ax.axhline(max(accuracies), color='red', ls='--', lw=1.5,
           label=f'Best: {max(accuracies)}%')
ax.legend(); ax.grid(axis='y', alpha=0.4); plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_accuracy_bar_chart.png"),
            dpi=150, bbox_inches='tight')
plt.close(); print("     Saved: 03_accuracy_bar_chart.png")

#  F1 / Precision / Recall 
metric_keys = ['F1-Score', 'Precision', 'Recall']
x, w = np.arange(len(model_names)), 0.25
fig, ax = plt.subplots(figsize=(14, 6))
for i, mk in enumerate(metric_keys):
    vals = [metrics[m][mk] for m in model_names]
    ax.bar(x + i*w, vals, w, label=mk, edgecolor='black', lw=0.6)
ax.set_xticks(x + w); ax.set_xticklabels(model_names, rotation=15)
all_metric_vals = [metrics[m][mk] for m in model_names for mk in metric_keys]
ax.set_ylim([max(0, min(all_metric_vals) - 8), 100])
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('F1-Score, Precision & Recall Comparison',
             fontsize=14, fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_f1_precision_recall.png"),
            dpi=150, bbox_inches='tight')
plt.close(); print("     Saved: 04_f1_precision_recall.png")

#  Confusion matrices 
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for i, (mname, y_pred) in enumerate(predictions.items()):
    ax = axes.flatten()[i]
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative','Positive'],
                yticklabels=['Negative','Positive'],
                ax=ax, lw=0.5, cbar=False)
    ax.set_title(f"{mname}\nAcc: {metrics[mname]['Accuracy']}%",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.suptitle('Confusion Matrices — All Models',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_confusion_matrices.png"),
            dpi=150, bbox_inches='tight')
plt.close(); print("     Saved: 05_confusion_matrices.png")

#  Improvement highlight
baseline = metrics['BiGRU']['Accuracy']
improved = metrics['BiGRU_Attention']['Accuracy']
diff     = improved - baseline
fig, ax  = plt.subplots(figsize=(8, 6))
bars = ax.bar(['BiGRU\n(Baseline)', 'BiGRU+Attention\n(Proposed)'],
              [baseline, improved],
              color=['#377eb8','#e41a1c'], edgecolor='black', width=0.45)
for bar, val in zip(bars, [baseline, improved]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f"{val}%", ha='center', fontsize=14, fontweight='bold')
y_lo = max(0, min(baseline, improved) - 10)
y_hi = min(100, max(baseline, improved) + 8)
ax.set_ylim([y_lo, y_hi])
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('BiGRU (Baseline) vs BiGRU+Attention (Proposed)',
             fontsize=13, fontweight='bold')
sign  = '+' if diff >= 0 else ''
color = 'green' if diff >= 0 else 'red'
ax.annotate(f'{sign}{diff:.2f}% vs baseline',
            xy=(1, improved),
            xytext=(0.3, (baseline + improved) / 2),
            fontsize=12, color=color, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "06_improvement_highlight.png"),
            dpi=150, bbox_inches='tight')
plt.close(); print("     Saved: 06_improvement_highlight.png")

# =============================================================================
# 7.  GENERATIVE AI COMPONENT
# =============================================================================
print("\n[6/8] Preparing text generation corpus ...")

word_index = imdb.get_word_index()
idx2word   = {v+3: k for k, v in word_index.items()}
idx2word.update({0:'pad', 1:'start', 2:'unk', 3:'unused'})

(raw_train, _), _ = imdb.load_data(num_words=VOCAB_SIZE)

corpus_words = []
for seq in raw_train[:4000]:
    words = [idx2word.get(i, 'unk') for i in seq if i > 3]
    corpus_words.append(' '.join(words))

raw_text = ' '.join(corpus_words)
raw_text = re.sub(r'[^a-z\s.,!?]', '', raw_text.lower())
raw_text = re.sub(r'\s+', ' ', raw_text).strip()
raw_text = raw_text[:150_000]          # 150k chars — fast but sufficient

print(f"      Corpus : {len(raw_text):,} characters")

chars      = sorted(set(raw_text))
char2idx   = {c: i for i, c in enumerate(chars)}
idx2char   = {i: c for c, i in char2idx.items()}
VOCAB_CHAR = len(chars)
print(f"      Vocab  : {VOCAB_CHAR} unique chars")

print("      Building windows ...")
X_gen, y_gen = [], []
for i in range(0, len(raw_text) - GEN_SEQ_LEN, GEN_STEP):
    X_gen.append([char2idx[c] for c in raw_text[i: i+GEN_SEQ_LEN]])
    y_gen.append(char2idx[raw_text[i + GEN_SEQ_LEN]])

X_gen = np.array(X_gen, dtype=np.float32) / VOCAB_CHAR
y_gen = tf.keras.utils.to_categorical(y_gen, num_classes=VOCAB_CHAR)
print(f"      Windows: {len(X_gen):,}")

print("\n[7/8] Training Character-Level LSTM Generator ...")
gen_model = Sequential([
    LSTM(GEN_LSTM_UNITS, input_shape=(GEN_SEQ_LEN, 1),
         return_sequences=True),
    Dropout(0.3),
    LSTM(GEN_LSTM_UNITS),
    Dropout(0.3),
    Dense(VOCAB_CHAR, activation='softmax')
], name="CharLSTM_Generator")

gen_model.compile(optimizer=Adam(5e-4), loss='categorical_crossentropy')
gen_model.summary()

X_gen_r = X_gen.reshape(len(X_gen), GEN_SEQ_LEN, 1)

gen_history = gen_model.fit(
    X_gen_r, y_gen,
    epochs=GEN_EPOCHS,
    batch_size=GEN_BATCH,
    callbacks=[
        EarlyStopping(monitor='loss', patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='loss', factor=0.5,
                          patience=4, verbose=1)
    ],
    verbose=1
)

#  Sampling 
def sample_temp(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_p = np.exp(preds - np.max(preds))
    preds = exp_p / exp_p.sum()
    return np.argmax(np.random.multinomial(1, preds, 1))

def generate_text(seed_text, temperature=0.7, length=GEN_LENGTH):
    seed = seed_text.lower()
    seed = seed.rjust(GEN_SEQ_LEN) if len(seed) < GEN_SEQ_LEN \
           else seed[-GEN_SEQ_LEN:]
    generated = ''
    for _ in range(length):
        x  = np.array([[char2idx.get(c, 0) for c in seed]],
                       dtype=np.float32) / VOCAB_CHAR
        pr = gen_model.predict(x.reshape(1, GEN_SEQ_LEN, 1),
                               verbose=0)[0]
        nc = idx2char[sample_temp(pr, temperature)]
        generated += nc
        seed = seed[1:] + nc
    return generated

#  Generate samples 
print("\n  Generated Movie Review Samples\n" + "-"*55)
seed = raw_text[300: 300+GEN_SEQ_LEN]

generated_samples = {}
for temp in GEN_TEMPS:
    text = generate_text(seed, temperature=temp, length=GEN_LENGTH)
    generated_samples[temp] = text
    print(f"\n  [Temperature = {temp}]")
    print(f"  {text}")

#  Generator loss plot 
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(gen_history.history['loss'], color='#e41a1c', lw=2.5)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Categorical Cross-Entropy Loss', fontsize=12)
ax.set_title('Character-Level LSTM Generator — Training Loss',
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "07_generator_loss.png"),
            dpi=150, bbox_inches='tight')
plt.close(); print("\n     Saved: 07_generator_loss.png")

#  Word frequency at each temperature 
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, temp in zip(axes, GEN_TEMPS):
    words = re.findall(r'[a-z]{2,}', generated_samples[temp])
    freq  = {}
    for w in words: freq[w] = freq.get(w, 0) + 1
    top10 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
    if top10:
        labels, counts = zip(*top10)
        ax.barh(list(labels)[::-1], list(counts)[::-1],
                color='#377eb8', edgecolor='black')
    ax.set_title(f'Temperature = {temp}\nTop-10 Generated Words',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Frequency'); ax.grid(axis='x', alpha=0.4)
plt.suptitle('Generated Text Word Analysis at Different Temperatures',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "08_generated_text_analysis.png"),
            dpi=150, bbox_inches='tight')
plt.close(); print("     Saved: 08_generated_text_analysis.png")

# =============================================================================
# 8.  SAVE FULL REPORT
# =============================================================================
print("\n[8/8] Saving metrics_report.txt ...")
with open(os.path.join(OUT_DIR, "metrics_report.txt"), 'w',
          encoding='utf-8') as f:
    f.write("=" * 65 + "\n")
    f.write("  NLP Comparative Analysis  [v3 FAST]\n")
    f.write("  Shiri et al. (2023) + BiGRU+Attention + Generative LSTM\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"{'Model':<22} {'Accuracy':>10} {'F1-Score':>10}"
            f" {'Precision':>11} {'Recall':>8}\n")
    f.write("-" * 65 + "\n")
    for mname, m in metrics.items():
        f.write(f"{mname:<22} {m['Accuracy']:>9}%  {m['F1-Score']:>9}%"
                f"  {m['Precision']:>9}%  {m['Recall']:>7}%\n")
    f.write("\n\nDetailed Classification Reports\n" + "=" * 65 + "\n")
    for mname, y_pred in predictions.items():
        f.write(f"\n--- {mname} ---\n")
        f.write(classification_report(y_test, y_pred,
                target_names=['Negative','Positive']))
    f.write("\n\n" + "=" * 65 + "\n")
    f.write("  GENERATIVE AI — Character-Level LSTM Output\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"Seed: {seed.strip()}\n\n")
    for temp, text in generated_samples.items():
        f.write(f"--- Temperature = {temp} ---\n{text}\n\n")
print("     Saved: metrics_report.txt")

# =============================================================================
# DONE
# =============================================================================
print("\n" + "=" * 65)
print("  ALL DONE! Here are the results: ")
print()
print("  01_accuracy_curves.png       02_loss_curves.png")
print("  03_accuracy_bar_chart.png    04_f1_precision_recall.png")
print("  05_confusion_matrices.png    06_improvement_highlight.png")
print("  07_generator_loss.png        08_generated_text_analysis.png")
print("  metrics_report.txt")
print("=" * 65 + "\n")