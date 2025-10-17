# src/infer.py
import numpy as np
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import pathlib

# --- Rutas ---
MODEL_PATH = pathlib.Path(__file__).resolve().parents[1] / "models" / "cnn_cifar10_final.keras"

# Clases de CIFAR-10
CLASS_NAMES = [
    "avión", "automóvil", "pájaro", "gato", "ciervo",
    "perro", "rana", "caballo", "barco", "camión"
]

_model_cache = None  # para no recargar cada vez


def load_model():
    """Carga el modelo entrenado y lo cachea en memoria."""
    global _model_cache
    if _model_cache is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"❌ No se encontró el modelo en {MODEL_PATH}")
        _model_cache = keras_load_model(MODEL_PATH)
        print(f"✅ Modelo cargado desde {MODEL_PATH}")
    return _model_cache


def get_class_names():
    """Devuelve la lista de nombres de clases CIFAR-10."""
    return CLASS_NAMES


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convierte imagen PIL a tensor normalizado listo para predecir."""
    image = image.resize((32, 32))  # CIFAR-10 es 32x32
    arr = img_to_array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)  # batch de 1
    return arr


def predict_proba(model, image: Image.Image) -> np.ndarray:
    """Devuelve las probabilidades para cada clase."""
    arr = preprocess_image(image)
    probs = model.predict(arr, verbose=0)[0]
    return probs

