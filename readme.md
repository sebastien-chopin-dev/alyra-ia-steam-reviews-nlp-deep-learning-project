# Projet 

Semaine 6 deep learning tensorflow

## Installation des dépendances

### For GPU users
pip install tensorflow[and-cuda]
### For CPU users
pip install tensorflow

### Verify installation

python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

### Dépendances développement

pip install keras tensorflow matplotlib jupyter pandas numpy scipy scikit-learn scikeras[tensorflow] optuna GPUtil seaborn gymnasium keras-nlp datasets kagglehub[pandas-datasets] kaggle

### Kaggle authentication

https://www.kaggle.com/docs/api#authentication

token at ~/.kaggle/kaggle.json

### API Docker build

cd api
docker-compose up --build

http://localhost:8000

### Scripts python

python -m src.reviews_en_finetuning_models

### TensorBoard

tensorboard --logdir=outputs/reports/bert_base_en_test/logs

### Stratégies d'entrainement

Phase 1 :  Tests rapide pour trouver les meilleures hyper paramétres:
- Sur un subset des reviews limité (50 000 au lieu de 200 000) test de plusieurs configurations 
de learning rate et de plusieurs architectures de couches de sortie finetuning. (12 configurations)
- Utilisation d'un modèle bert plus léger 
- "bert_small_en_uncased" 29 M de paramétres.
(Environ 2 minutes par entrainement avec RTX 4070)

Phase 2: Prendre les 3 meilleures combinaisons d'hyper paramétres et tests sur modèles standard
Subset plus large (100000 reviews)
- "distil_bert_base_en_uncased" 66M de params
- "bert_base_en_uncased" 110M de params

Phase 3: Prendre la meilleure combinaison d'hyper paramétre et tests sur modèles plus performant:
Subset complet (200000 reviews)
- "bert_base_en_uncased" 110M de params
- "roberta_base_en" 125M de params
- "deberta_v3_small_en" 86M



