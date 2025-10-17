from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.model_cnn import build_cnn

DATA_PATH = Path("data/processed/cifar10_preprocesado.npz")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports/figures")

def plot_history(history, out_prefix="cifar10_cnn"):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    # accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.legend()
    plt.title("Accuracy")
    plt.savefig(REPORTS_DIR / f"{out_prefix}_accuracy.png")
    plt.close()
    # loss
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.title("Loss")
    plt.savefig(REPORTS_DIR / f"{out_prefix}_loss.png")
    plt.close()
    print("📊 Gráficas de entrenamiento guardadas.")

def main():
    # cargar datos
    data = np.load(DATA_PATH)
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]
    print(f"✔️ Dataset cargado: x_train={x_train.shape}, y_train={y_train.shape}")

    # construir modelo
    model = build_cnn()
    model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=["accuracy"])
    model.summary()

    # callbacks
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS_DIR / "cnn_cifar10_best.keras"
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, save_best_only=True)
    ]

    print("🚀 Iniciando entrenamiento...")
    history = model.fit(x_train, y_train,
                        validation_split=0.1,
                        epochs=10,
                        batch_size=128,
                        callbacks=callbacks,
                        verbose=1)

    plot_history(history)

    # evaluación
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"📊 Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    # guardar modelo final
    final_path = MODELS_DIR / "cnn_cifar10_final.keras"
    model.save(final_path)
    print(f"💾 Modelo final guardado en {final_path}")

if __name__ == "__main__":
    main()



