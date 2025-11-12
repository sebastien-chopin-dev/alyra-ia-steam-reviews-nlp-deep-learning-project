import numpy as np


def calculate_optimal_batch_size(model, sample_input_shape, gpu_memory_gb=8):
    """
    Calcule la taille de batch optimale basée sur la mémoire disponible
    """
    # Estimation de la mémoire par échantillon (en bytes)
    input_memory = np.prod(sample_input_shape) * 4  # float32 = 4 bytes

    # Estimation de la mémoire du modèle
    total_params = model.count_params()
    model_memory = total_params * 4 * 3  # Poids + gradients + momentum

    # Mémoire disponible (en bytes)
    available_memory = gpu_memory_gb * 1024**3 * 0.8  # 80% de la mémoire GPU

    # Calcul du batch size optimal
    memory_per_sample = input_memory * 2  # Forward + backward pass
    usable_memory = available_memory - model_memory
    optimal_batch_size = int(usable_memory / memory_per_sample)

    # Arrondir à la puissance de 2 la plus proche
    return 2 ** int(np.log2(optimal_batch_size))


# Utilisation
# optimal_bs = calculate_optimal_batch_size(model, (224, 224, 3))
# print(f"Batch size optimal recommandé: {optimal_bs}")
