from keras.callbacks import Callback
import psutil
import GPUtil


def monitor_training_resources():
    """Surveillance des ressources pendant l'entraînement"""

    class ResourceMonitor(Callback):

        def __init__(self):
            super().__init__()
            self.memory_usage = []
            self.gpu_usage = []

        def on_epoch_end(self, epoch, logs=None):
            # Mémoire RAM
            ram_percent = psutil.virtual_memory().percent

            # Utilisation GPU
            gpus = GPUtil.getGPUs()
            gpu_percent = gpus[0].memoryUtil * 100 if gpus else 0

            self.memory_usage.append(ram_percent)
            self.gpu_usage.append(gpu_percent)

            print(
                f"Epoch {epoch+1} - RAM: {ram_percent:.1f}% - GPU: {gpu_percent:.1f}%"
            )

    return ResourceMonitor()


# # Utilisation pendant l'entraînement
# monitor = monitor_training_resources()
# model.fit(X_train, y_train, batch_size=64, epochs=100, callbacks=[monitor])
