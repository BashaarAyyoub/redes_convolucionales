# src/data_prep.py
from pathlib import Path
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def load_raw():
    """Carga CIFAR-10 sin modificar."""
    return cifar10.load_data()

def preprocess(x_train, y_train, x_test, y_test):
    """Normaliza a [0,1] y convierte etiquetas a one-hot."""
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0
    y_train_oh = to_categorical(y_train, num_classes=10)
    y_test_oh  = to_categorical(y_test,  num_classes=10)
    return x_train, y_train_oh, x_test, y_test_oh

def save_preprocessed(x_train, y_train, x_test, y_test,
                      out_path="data/processed/cifar10_preprocesado.npz"):
    """Guarda el dataset ya preprocesado en .npz comprimido."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        x_train=x_train, y_train=y_train,
        x_test=x_test,   y_test=y_test
    )
    print(f"✅ Dataset preprocesado guardado en {Path(out_path).resolve()}")

__all__ = ["CLASS_NAMES", "load_raw", "preprocess", "save_preprocessed"]




