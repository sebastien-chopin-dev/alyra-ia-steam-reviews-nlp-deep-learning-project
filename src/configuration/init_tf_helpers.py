import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import keras
import keras_nlp


def init_gpu_for_tf():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU(s) détecté(s): {len(gpus)} - Croissance mémoire activée")
        else:
            print("⚠️  Aucun GPU détecté - Utilisation du CPU")
    except Exception as e:
        print(f"Configuration GPU: {e}")


def init_graph_plt():
    # Configuration graphiques
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")


def init_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def show_tf_keras_version_engine():
    print(f"\nTensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    print(f"KerasNLP version: {keras_nlp.__version__}")
