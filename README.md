# CENG467 Term Project - Unsupervised Neural Machine Translation

**Group 9:** Ali Alp Harac and Ihsan Yagiz Sakizlioglu

This repository implements an unsupervised Turkish-English translation
pipeline using a shared Transformer, denoising autoencoder (DAE), and
iterative back-translation (IBT).

## Pipeline

- CC-100 monolingual data download and cleaning
- Joint 32K SentencePiece vocabulary
- Shared 4-layer Transformer encoder-decoder
- Turkish and English DAE pretraining
- Bidirectional iterative back-translation
- FLORES-200 BLEU and chrF evaluation
- Gradio translation demo

Python 3.8 or newer is supported.

## Google Colab

```python
from google.colab import drive
drive.mount("/content/drive")

!git clone https://github.com/alialp5959/ceng467-term-project.git
%cd ceng467-term-project
!pip install -r requirements.txt
```

### 1. Preprocess

```bash
python src/preprocess.py --step all
```

### 2. Train the DAE

```bash
python src/train_autoencoder.py
```

The DAE checkpoint is saved as:

```text
/content/drive/MyDrive/CENG467_Project/checkpoints/checkpoint_latest.pt
```

### 3. Run corrected IBT

```bash
python src/backtranslate.py \
  --checkpoint /content/drive/MyDrive/CENG467_Project/checkpoints/checkpoint_latest.pt \
  --iterations 10 \
  --force-regenerate
```

`--force-regenerate` is recommended for the first run after the
target-language-prefix fix. Later interrupted runs can reuse verified caches.

IBT checkpoints are saved separately:

```text
checkpoint_ibt_iter1.pt
checkpoint_ibt_iter2.pt
...
checkpoint_ibt_iter10.pt
checkpoint_ibt_latest.pt
```

Resume an interrupted run with:

```bash
python src/backtranslate.py \
  --checkpoint /content/drive/MyDrive/CENG467_Project/checkpoints/checkpoint_ibt_latest.pt \
  --iterations 10
```

### 4. Evaluate the IBT model

Do not evaluate `checkpoint_latest.pt`; that file is the DAE-only checkpoint.

```bash
python src/evaluate.py \
  --checkpoint /content/drive/MyDrive/CENG467_Project/checkpoints/checkpoint_ibt_iter10.pt \
  --strategy beam
```

### 5. Launch the demo

```bash
python src/demo.py \
  --checkpoint /content/drive/MyDrive/CENG467_Project/checkpoints/checkpoint_ibt_iter10.pt
```

## Important Training Fixes

The current implementation:

- prefixes translation inputs with the **target** language token;
- uses `<TR>` for synthetic-English to real-Turkish training;
- uses `<EN>` for synthetic-Turkish to real-English training;
- mixes DAE loss into every IBT iteration;
- samples different monolingual sentences each iteration;
- keeps optimizer state when resuming IBT;
- rejects stale synthetic caches using model and sample fingerprints;
- evaluates explicit IBT checkpoints;
- uses EOS-aware beam search with length normalization.

## Results (Corrected Pipeline)

Final evaluation on FLORES-200 devtest (1012 sentences per direction):

| Model | Decoding | TR→EN BLEU | TR→EN chrF | EN→TR BLEU | EN→TR chrF |
|---|---|---:|---:|---:|---:|
| Copy Baseline | — | 2.82 | 21.16 | 2.83 | 20.02 |
| MUSE Word-by-Word | — | 3.25 | 34.63 | 2.46 | 36.00 |
| **Our UNMT (Iter6 200k)** | Beam 4 | **2.63** | 23.07 | 2.41 | 20.79 |
| Our UNMT (Iter7 Filtered) | Greedy | 2.49 | 22.84 | **2.50** | 20.85 |
| mBART-50 (Zero-Shot) | Beam 4 | 30.25 | 58.01 | 18.10 | 52.05 |
| Helsinki-NLP (Supervised) | Beam 4 | 30.21 | 58.97 | 31.08 | 61.50 |

## Historical Results (Pre-Bugfix)

These scores belong to the original run before target-language conditioning
and EOS leakage were fixed:

| Model | TR→EN BLEU | EN→TR BLEU |
|---|---:|---:|
| Original UNMT, 1 IBT | 0.09 | 0.10 |
| Original UNMT, 10 IBT | 0.17 | 0.04 |
