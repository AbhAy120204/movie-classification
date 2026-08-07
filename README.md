# Movie Genre Classification (Multi-Label)

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface)](https://huggingface.co/Abhay-learns/distilbert-genre)
[![Gradio](https://img.shields.io/badge/Demo-Gradio-orange?logo=gradio)](https://huggingface.co/spaces/Abhay-learns/movie-genre-classifier-demo)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**[Live Demo →](https://huggingface.co/spaces/Abhay-learns/movie-genre-classifier-demo)**

Predicting movie genres from plot descriptions using NLP. This project compares a classical TF-IDF + Logistic Regression baseline against a fine-tuned DistilBERT transformer model on 182k IMDB movies across 20 genres.

---

## Results

| Model | Micro F1 | Macro F1 | Jaccard | ROC-AUC |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.4709 | 0.4032 | 0.3079 | 0.7717 |
| DistilBERT-base-uncased | **0.5359** | **0.4561** | **0.3660** | **0.9143** |

> DistilBERT fine-tuned for 5 epochs on 145k samples — improves Micro F1 by +6.5% and ROC-AUC by +14.3% over the classical baseline.

---

## Project Structure

```
movie-classification/
├── dataset/              # 16 genre CSVs from Kaggle (182k movies after dedup)
├── notebooks/
│   ├── 01_EDA.ipynb                    # Genre distribution, co-occurrence, word clouds
│   ├── 02_baseline_classical_ml.ipynb  # TF-IDF + Logistic Regression baseline
│   └── 03_llm_finetuning.ipynb         # DistilBERT fine-tuning + HuggingFace Hub push
├── app/
│   └── app.py            # Gradio demo app
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Data & EDA | pandas, matplotlib, seaborn, wordcloud |
| Classical ML | scikit-learn (TF-IDF, Logistic Regression) |
| Deep Learning | PyTorch, HuggingFace Transformers |
| Fine-tuning | DistilBERT, HuggingFace Trainer API |
| Experiment Tracking | Weights & Biases (W&B) |
| Deployment | Gradio, HuggingFace Spaces & Hub |

---

## Dataset

- **Source:** [IMDB Movies Dataset Based on Genre](https://www.kaggle.com/datasets/rajugc/imdb-movies-dataset-based-on-genre) (Kaggle)
- **Raw:** ~368k rows across 16 CSV files
- **After cleaning:** 182k unique movies, 20 genres (genres with < 1000 samples dropped)
- **Split:** MultilabelStratifiedKFold — ensures each genre is proportionally represented in train and test

---

## How to Run

```bash
git clone https://github.com/AbhAy120204/movie-classification.git
cd movie-classification
pip install -r requirements.txt
```

Run notebooks in order:
1. `notebooks/01_EDA.ipynb` — data exploration (runs locally)
2. `notebooks/02_baseline_classical_ml.ipynb` — classical ML baseline (runs locally)
3. `notebooks/03_llm_finetuning.ipynb` — LLM fine-tuning (requires GPU)

Run the Gradio app locally:
```bash
cd app
python app.py
```

---

## Model

Fine-tuned model available on HuggingFace Hub:
**[Abhay-learns/distilbert-genre](https://huggingface.co/Abhay-learns/distilbert-genre)**
