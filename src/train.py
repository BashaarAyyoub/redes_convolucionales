# src/train.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.model_cnn import build_cnn

# Rutas
DATA_PATH   = Path("data/processed/cifar10_preprocesado.npz")
MODELS_DIR  = Path("models")
REPORTS_DIR = Path("reports/figures")


def plot_history(history, out_prefix="cifar10_cnn"):
    """Guarda las curvas de accuracy y loss (train/val) en reports/figures/."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Accuracy
    plt.figure()
    plt.plot(history.history.get('accuracy', []), label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy')
    acc_path = REPORTS_DIR / f"{out_prefix}_accuracy.png"
    plt.savefig(acc_path, dpi=150, bbox_inches="tight")
    print(f"✅ Guardada {acc_path}")

    # Loss
    plt.figure()
    plt.plot(history.history.get('loss', []), label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss')
    loss_path = REPORTS_DIR / f"{out_prefix}_loss.png"
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    print(f"✅ Guardada {loss_path}")


def main():
    print("🚀 Iniciando entrenamiento...")

    # 1) Cargar datos preprocesados (Fase 1 debe haberse ejecutado antes)
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No encuentro {DATA_PATH}. "
            f"Ejecuta primero:  python -m src.train_cnn"
        )
    data = np.load(DATA_PATH, allow_pickle=True)
    x_train, y_train = data["x_train"], data["y_train"]
    x_test,  y_test  = data["x_test"],  data["y_test"]
    print(f"✔️ Dataset cargado: x_train={x_train.shape}, y_train={y_train.shape}")

    # 2) Modelo
    model = build_cnn(input_shape=x_train.shape[1:], num_classes=y_train.shape[1])
    model.compile(optimizer=Adam(),
                  loss=CategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.summary()

    # 3) Callbacks y rutas de salida
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS_DIR / "cnn_cifar10_best.keras"
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        ModelCheckpoint(filepath=ckpt_path, save_best_only=True, monitor="val_accuracy")
    ]

    # 4) Entrenamiento
    history = model.fit(
        x_train, y_train,
        epochs=10,             # >=10 como pide el enunciado
        batch_size=128,
        validation_split=0.1,  # 10% para validación
        callbacks=callbacks,
        verbose=1
    )

    # 5) Curvas
    plot_history(history, out_prefix="cifar10_cnn")

    # 6) Evaluación en test
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"📊 Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    # 7) Guardar modelo final
    final_path = MODELS_DIR / "cnn_cifar10_final.keras"
    model.save(final_path)
    print(f"💾 Modelo final guardado en {final_path}")


if __name__ == "__main__":
    main()


