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

pip install keras tensorflow matplotlib jupyter pandas numpy scipy scikit-learn scikeras[tensorflow] optuna GPUtil seaborn gymnasium keras-nlp



