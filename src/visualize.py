import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from src.data_prep import CLASS_NAMES

def plot_samples_grid(x, y, out_path="reports/figures/cifar10_muestras.png", n=25):
    """Muestra una cuadrícula de imágenes con etiquetas."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,6))
    idxs = np.random.choice(len(x), n, replace=False)
    for i, idx in enumerate(idxs):
        plt.subplot(5,5,i+1)
        plt.imshow(x[idx])
        plt.title(CLASS_NAMES[int(y[idx])])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"📸 Figura guardada en {out_path}")
