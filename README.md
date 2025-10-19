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
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m src.app

El motivo de que te pida python -m src.app y no simplemente python src/app.py es por cómo funciona Python con los paquetes y módulos. Te explico claro y rápido:

🔹 Caso 1: python src/app.py

Aquí Python cree que src es solo una carpeta normal, no un paquete.

Entonces cuando dentro de app.py haces:

from src.infer import load_model


Python no sabe dónde está src → te lanza ModuleNotFoundError: No module named 'src'.

🔹 Caso 2: python -m src.app

Con la opción -m, le dices a Python:

"Ejecuta el módulo app que está dentro del paquete src".

Así, src se trata como un paquete Python, y ya entiende perfectamente imports como:

from src.infer import load_model
from src.data_prep import preprocess

📌 En resumen:

Si ejecutas desde dentro de la carpeta src/, puedes hacer:

python app.py


(pero ahí ya no funcionan igual las rutas a templates/ y static/).

Lo más limpio es siempre ejecutar desde la raíz del proyecto:

python -m src.app

