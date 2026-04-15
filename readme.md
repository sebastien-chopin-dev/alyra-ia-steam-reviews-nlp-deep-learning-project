# Steam Reviews Sentiment Analysis — NLP Deep Learning  — BERT Finetuning

**Alyra Project — Deep Learning**

Binary sentiment classification (positive / negative) of Steam game reviews using domain-specific BERT fine-tuning with Keras NLP and TensorFlow.

---

## Overview

Most general-purpose sentiment models are trained on movie or product reviews and struggle with gaming-specific language (e.g. "this game is cancer", "absolute banger", "pay-to-win garbage"). This project fine-tunes BERT on 230k+ authentic Steam reviews to build a **specialized gaming sentiment classifier**.

**Key results:**
- Baseline (VADER lexical): ~70% accuracy
- Generic DistilBERT (SST-2 movie reviews): ~85% accuracy
- **Our model (BERT fine-tuned on Steam)**: **91–93% accuracy**

---

## Dataset

[Steam Reviews — Kaggle (100 million reviews, 40GB)](https://www.kaggle.com/datasets/kieranpoc/steam-reviews/data)

The raw dataset is filtered and processed before fine-tuning:
- Language filter: English and French
- Quality filter: `weighted_vote_score > 0.6` (Steam's credibility metric)
- Authenticity filter: verified purchase, not received for free, >30 min playtime
- Length filter: 10–400 words (fits BERT's 512-token limit)
- Cleaning: HTML/BBCode tags removed, ASCII art filtered out
- Class balancing: 50/50 positive/negative undersampling

**Final datasets:**
- English: ~229,000 reviews
- French: ~60,000 reviews

---

## Models

| Model | Preset | Accuracy | Use |
|---|---|---|---|
| BERT Small | `bert_small_en_uncased` | ~90% | Production (lightweight) |
| BERT Base | `bert_base_en_uncased` | ~92% | Production |
| DistilBERT Base | `distil_bert_base_en_uncased` | ~93% | Production (best) |

Trained models are saved in `outputs/models/prod/`.

---

## Project Structure

```
├── notebooks/
│   ├── 1_reviews_dataset_exploration.ipynb        # Dataset exploration and filter strategy
│   ├── 2_create_fine_tuning_reviews_dataset_en_fr.ipynb  # Dataset creation pipeline
│   ├── 3_fine_tuning_reviews_en_preset_keras_nlp_bert.ipynb  # BERT fine-tuning baseline (POC)
│   ├── 4_fine_tuning_phases_strategy.ipynb        # Multi-phase training strategy
│   ├── compare_prediction_with_vader.ipynb        # Model comparison vs VADER
│   └── 5_deployment_api_docker.ipynb              # API deployment documentation
│
├── src/scripts/
│   ├── dataset/
│   │   ├── create_all_review_fr_file.py           # Extract all French reviews
│   │   ├── create_weighted_score_06_review_en_file.py  # Extract English reviews (score > 0.6)
│   │   ├── create_reviews_dataset_en_fr_for_fine_tuning.py  # Build final EN/FR datasets
│   │   └── run_complete_datasets_prep_for_fine_tuning.py    # Run full dataset pipeline
│   ├── fine_tuning/
│   │   ├── run_phase1_fast_fine_tuning_to_find_best_hyperparams.py
│   │   ├── run_phase2_small_fine_tuning.py
│   │   ├── run_phase3_base_and_distill_fine_tuning.py
│   │   └── run_phase3_1_base_and_distill_fine_tuning.py
│   ├── run_complete_reviews_en_bert_fine_tuning.py  # Full English pipeline
│   └── run_complete_reviews_fr_bert_fine_tuning.py  # Full French pipeline
│
├── api/
│   ├── app/main.py          # FastAPI application
│   ├── Dockerfile
│   ├── docker-compose.yaml
│   └── requirements.txt
│
└── outputs/
    ├── models/
    │   ├── poc/             # Prototype models
    │   └── prod/            # Production models (.keras)
    ├── reports/             # TensorBoard logs
    └── evaluation_results_en_phase*.csv  # Per-phase metrics
```

---

## Training Strategy (4 Phases)

### Phase 1 — Hyperparameter search (small model, reduced dataset)
Grid search over learning rates, output layer architectures, and callback strategies on 50,000 reviews with `bert_small_en_uncased`.

```bash
python -m src.scripts.fine_tuning.run_phase1_fast_fine_tuning_to_find_best_hyperparams
```

### Phase 2 — Validate best hyperparameters (small model, full dataset)
Take the top 6 hyperparameter combinations from Phase 1 and train on the full 229k English dataset.

```bash
python -m src.scripts.fine_tuning.run_phase2_small_fine_tuning
```

### Phase 3 — Train base models (full dataset)
Train `bert_base_en_uncased` and `distil_bert_base_en_uncased` with the best hyperparameters.

```bash
python -m src.scripts.fine_tuning.run_phase3_base_and_distill_fine_tuning
```

### Phase 3.1 — Sequence length experiments
Test sequence lengths of 128, 256, and 512 on DistilBERT to evaluate the accuracy/speed trade-off.

```bash
python -m src.scripts.fine_tuning.run_phase3_1_base_and_distill_fine_tuning
```

**Accuracy progression:**

| Phase | Model | Dataset size | Accuracy |
|---|---|---|---|
| Baseline | BERT Small | 20,000 | ~0.82 |
| Phase 1 | BERT Small | 50,000 | ~0.88 |
| Phase 2 | BERT Small | 229,000 | ~0.90 |
| Phase 3 | BERT Base | 229,000 | ~0.92 |
| Phase 3.1 | DistilBERT (seq=256) | 229,000 | ~0.93 |

---

## Full Pipeline (shortcut)

```bash
# Step 1 — Prepare datasets
python -m src.scripts.dataset.run_complete_datasets_prep_for_fine_tuning

# Step 2 — Full English fine-tuning pipeline
python -m src.scripts.run_complete_reviews_en_bert_fine_tuning

# Step 3 — Full French fine-tuning pipeline
python -m src.scripts.run_complete_reviews_fr_bert_fine_tuning
```

---

## API Deployment

The sentiment prediction API is built with FastAPI and containerized with Docker.

```bash
cd api

# First build
docker-compose up --build

# Subsequent launches
docker-compose up
```

- API: http://localhost:8000
- Docs: http://localhost:8000/docs

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This game is absolutely amazing, 100 hours and still going!"}'
```

**Example response:**
```json
{
  "sentiment": "POSITIVE",
  "confidence": 99.99,
  "probabilities": { "negative": 0.01, "positive": 99.99 }
}
```

---

## Installation

### Requirements

- Deep learning experiment requires a graphics card with a significant amount of VRAM. 
- (> 10 GB VRAM minimum and > 16 GB VRAM recommended)
- I made my expriment with an Nvidia RTX 4070 (12gb VRAM). 

```bash
# GPU (recommended)
pip install tensorflow[and-cuda]

# CPU only
pip install tensorflow
```

```bash
# All dependencies
pip install keras tensorflow matplotlib jupyter pandas numpy scipy \
  scikit-learn scikeras[tensorflow] optuna GPUtil seaborn \
  keras-nlp datasets kagglehub[pandas-datasets] vaderSentiment

# or with the requirments.txt
pip install -r requirements.txt
```

### Verify TensorFlow + GPU

```bash
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Kaggle Authentication

Required to download the dataset.
See: https://www.kaggle.com/docs/api#authentication
Place your token at `~/.kaggle/kaggle.json`

---

## Monitoring

```bash
tensorboard --logdir=outputs/reports/bert_base_en_test/logs
```
