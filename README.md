# CENG467 Term Project - Unsupervised Neural Machine Translation (UNMT)

**Group 9**: [Your Name/ID Here]

This repository contains the complete implementation of our Unsupervised Neural Machine Translation (UNMT) system for Turkish-English, trained without any parallel data using Denoising Autoencoder (DAE) and Iterative Back-Translation (IBT), following the methodology of Lample et al. (2018).

## Features

- **Full Pipeline**: Data downloading, cleaning, joint SentencePiece vocabulary training, model training, and evaluation.
- **Lightweight Architecture**: 4-layer Transformer Encoder-Decoder (46M parameters) optimized for Google Colab T4.
- **Denoising Autoencoder (DAE)**: Word shuffle, dropout, and mask noise generation.
- **Iterative Back-Translation (IBT)**: Synthetic parallel data generation and alternating cross-entropy/DAE training.
- **Comprehensive Evaluation**: Automated BLEU and chrF scoring on FLORES-200.
- **Interactive Demo**: Gradio-based web interface for testing translations live.

## Quick Start (Google Colab)

The project is designed to run seamlessly on Google Colab, leveraging Google Drive for data storage and checkpointing.

1. **Mount Drive & Clone Repository**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
   !git clone https://github.com/alialp5959/ceng467-term-project.git
   %cd ceng467-term-project
   ```

2. **Install Dependencies**:
   ```python
   !pip install -r requirements.txt
   ```

3. **Run Preprocessing Pipeline** (Downloads CC-100, cleans it, and trains SentencePiece):
   ```python
   !python src/preprocess.py --step all
   ```

4. **Train Denoising Autoencoder (DAE)**:
   ```python
   !python src/train_autoencoder.py
   ```

5. **Iterative Back-Translation (IBT)**:
   ```python
   !python src/backtranslate.py --checkpoint /content/drive/MyDrive/CENG467_Project/checkpoints/checkpoint_latest.pt --iterations 3
   ```

6. **Evaluate on FLORES-200**:
   ```python
   !python src/evaluate.py --checkpoint /content/drive/MyDrive/CENG467_Project/checkpoints/checkpoint_latest.pt
   ```

7. **Launch Live Demo**:
   ```python
   !python src/demo.py --checkpoint /content/drive/MyDrive/CENG467_Project/checkpoints/checkpoint_latest.pt
   ```

## Repository Structure

- `configs/`: YAML configuration files.
- `src/`: Core implementation scripts.
  - `preprocess.py`: CC-100 dataset download, filtering, and tokenization.
  - `model.py`: Transformer architecture.
  - `noise.py`: DAE noise functions (shuffle, dropout, mask).
  - `dataset.py`: Dataloaders for monolingual and synthetic parallel data.
  - `train_autoencoder.py`: DAE training loop.
  - `generate.py`: Greedy and Beam Search decoding.
  - `backtranslate.py`: IBT training loop.
  - `evaluate.py`: BLEU/chrF evaluation on FLORES-200.
  - `error_analysis.py`: Sample generator for manual qualitative review.
  - `demo.py`: Gradio interface.
- `report/`: LaTeX source for the final project report.

## Baseline Results (Checkpoint)
- **Copy Baseline**: TR->EN: 2.82 BLEU, EN->TR: 2.83 BLEU
- **Word-by-Word MUSE Baseline**: ~1-2 BLEU
- **Helsinki-NLP (Supervised Reference)**: TR->EN: 30.21 BLEU, EN->TR: 31.08 BLEU
