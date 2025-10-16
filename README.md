# redes_convolucionales
# CNN CIFAR-10 (Fase 1)

Primer commit: **carga, exploración y preprocesado** del dataset CIFAR-10 para una CNN en TensorFlow/Keras.

## Qué hace
- Carga CIFAR-10
- Visualiza muestras de las 10 clases (sin aplanar, RGB)
- Normaliza a `[0,1]`
- Convierte etiquetas a **one-hot**
- Guarda un `.npz` con los datos preprocesados en `data/processed/`

## Cómo ejecutar
```bash
pip install -r requirements.txt
python src/train_cnn.py
