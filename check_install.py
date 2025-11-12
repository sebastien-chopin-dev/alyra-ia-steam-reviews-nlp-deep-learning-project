import sys
import subprocess
import tensorflow as tf
import keras

# Vérifier les GPU physiques
result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
print(result.stdout)

# Lister tous les périphériques
print("Devices:", tf.config.list_physical_devices())

# GPU spécifiquement
gpus = tf.config.list_physical_devices("GPU")
print(f"Nombre de GPU: {len(gpus)}")

for gpu in gpus:
    print(f"GPU: {gpu}")

#  Test d'importation
print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
print("Python version:", sys.version)
print("Python executable:", sys.executable)

# Vérification compatibilité
print("Version détaillée:")
print(tf.version.VERSION)
print("Git version:", tf.version.GIT_VERSION)
print("Compilateur:", tf.version.COMPILER_VERSION)

# Croissance mémoire progressive (recommandé)
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Test création tenseur
hello = tf.constant("Hello, TensorFlow!")
print(hello)
# Test opération simple
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
result = tf.add(a, b)
print("Addition:", result.numpy())


# Test calcul sur GPU
# Vérifier placement automatique
with tf.device("/GPU:0"):
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)
    print("Calcul GPU réussi:", c.shape)
