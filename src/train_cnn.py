from src.data_prep import load_raw, preprocess, save_preprocessed, CLASS_NAMES
from src.visualize import plot_samples_grid

def main():
    (x_train, y_train), (x_test, y_test) = load_raw()
    print(f"x_train_raw: {x_train.shape}, y_train_raw: {y_train.shape}")
    print(f"x_test_raw : {x_test.shape}, y_test_raw : {y_test.shape}")

    # Visualizar ejemplos
    plot_samples_grid(x_train, y_train)

    # Preprocesar
    x_train_p, y_train_p, x_test_p, y_test_p = preprocess(x_train, y_train, x_test, y_test)
    print("✔️ Preprocesado completado")
    print(f"x_train rango: min={x_train_p.min():.3f} max={x_train_p.max():.3f}")
    print(f"y_train one-hot shape: {y_train_p.shape} | y_test one-hot shape: {y_test_p.shape}")

    # Guardar
    save_preprocessed(x_train_p, y_train_p, x_test_p, y_test_p)

if __name__ == "__main__":
    main()
