# src/train_cnn.py
from pathlib import Path
import numpy as np

from src.data_prep import (
    load_raw, load_and_preprocess, save_preprocessed, CLASS_NAMES
)
from src.visualize import plot_samples_grid

def main():
    # 1) Cargar datos crudos para visualizar por clase
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = load_raw()
    print(f"x_train_raw: {x_train_raw.shape}, y_train_raw: {y_train_raw.shape}")
    print(f"x_test_raw : {x_test_raw.shape},  y_test_raw : {y_test_raw.shape}")

    # 2) Visualizar muestras (sin aplanar, RGB)
    y_indices = y_train_raw.reshape(-1)  # (N,1) -> (N,)
    fig_path = Path("reports/figures/cifar10_muestras.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_samples_grid(
        x=x_train_raw,
        y_indices=y_indices,
        class_names=CLASS_NAMES,
        n_per_class=5,
        save_path=fig_path
    )

    # 3) Preprocesar: normalización + one-hot
    (x_train, y_train), (x_test, y_test), class_names = load_and_preprocess()
    print("✔️ Preprocesado completado")
    print(f"x_train rango: min={x_train.min():.3f} max={x_train.max():.3f}")
    print(f"y_train one-hot shape: {y_train.shape} | y_test one-hot shape: {y_test.shape}")

    # 4) Guardar dataset preprocesado
    save_preprocessed(((x_train, y_train), (x_test, y_test), class_names))

if __name__ == "__main__":
    main()

