# Projet IA - Prédiction sur les marketplaces des consoles digitales

## Alyra Bloc 05 - Deep learning

Ce projet consiste a utiliser un algorithme de deep learning pour classifier automatiquement le sentiment des commentaires de joueurs. Binaire (positif ou négatif).

## Positionnement du projet

- Fine-tuning spécifique sur les reviews de jeux vidéo Steam
- Spécialisation sur le langage et les expressions propres à la communauté du gaming
- Utilisation de modèles BERT via keras NLP Tensorflow (Tiny, small, DistilBERT)

## Dataset 

Steam Reviews Dataset (Kaggle, 100 millions reviews) :

https://www.kaggle.com/datasets/kieranpoc/steam-reviews/data

## Installation des dépendances

### For GPU users
pip install tensorflow[and-cuda]
### For CPU users
pip install tensorflow

### Verify installation

python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

### Dépendances développement

pip install keras tensorflow matplotlib jupyter pandas numpy scipy scikit-learn scikeras[tensorflow] optuna GPUtil seaborn gymnasium keras-nlp datasets kagglehub[pandas-datasets] vaderSentiment

### Kaggle authentication

https://www.kaggle.com/docs/api#authentication

token at ~/.kaggle/kaggle.json

## Structure du projet

### Notebooks

#### 1_reviews_dataset_exploration.ipynb
#### 2_create_fine_tuning_reviews_dataset_en_fr.ipynb
#### 3_fine_tuning_reviews_en_preset_keras_nlp_bert.ipynb
#### 4_fine_tuning_phases_strategy.ipynb
#### 5_deployment_api_docker copy.ipynb

### Scripts python

Fichiers python disponible dans src/scripts

#### dataset
#### fine_tuning
#### run_complete

### Déploiement API Docker build

cd api
docker-compose up --build

http://localhost:8000

### Scripts python

python -m src.reviews_en_finetuning_models

### TensorBoard

tensorboard --logdir=outputs/reports/bert_base_en_test/logs



