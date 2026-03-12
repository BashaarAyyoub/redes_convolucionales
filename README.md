# 🧠 Image Classification with Convolutional Neural Networks (CNN)

This repository contains the project **redes_convolucionales**, which implements an **image classification system using Convolutional Neural Networks (CNNs)**.

The project uses the **CIFAR-10 dataset** and is implemented using **TensorFlow/Keras**. It also includes a **Flask-based web interface** that allows users to run the model and test predictions with images.

Repository:
https://github.com/BashaarAyyoub/redes_convolucionales

---

# 📌 Project Overview

This project builds a **deep learning model capable of classifying images into different categories** using a Convolutional Neural Network.

The current phase of the project focuses on:

* Loading the **CIFAR-10 dataset**
* Exploring the dataset
* Visualizing sample images from the 10 classes
* Preprocessing the data
* Preparing the dataset for CNN training

---

# 📦 Dataset

The project uses the **CIFAR-10 dataset**, which contains:

* **60,000 images**
* Image size of **32x32 pixels**
* **10 different classes**

The classes included in the dataset are:

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

---

# ⚙️ What the Code Does

The current pipeline performs the following steps:

1. Loads the CIFAR-10 dataset
2. Visualizes sample images from the 10 classes (RGB images without flattening)
3. Normalizes image pixel values to the **[0,1] range**
4. Converts labels to **one-hot encoding**
5. Saves the processed dataset in a `.npz` file inside:

```
data/processed/
```

This allows the preprocessed data to be reused without repeating the entire preprocessing pipeline.

---

# 📂 Project Structure

```
redes_convolucionales
│
├── src/                     # Main project code
│   ├── app.py               # Main execution script
│   ├── infer.py             # Model inference
│   ├── data_prep.py         # Data preprocessing
│
├── templates/               # Flask HTML templates
├── static/                  # Static files
├── reports/figures          # Generated visualizations
│
├── requirements.txt         # Project dependencies
├── README.md
└── Redes_Convolucionales.ipynb   # Experiment notebook
```

---

# 🚀 How to Run the Project

It is recommended to run the project **from the root directory of the repository**.

### 1️⃣ Activate the virtual environment

```
.\.venv\Scripts\activate
```

If the environment does not exist, create it with:

```
python -m venv .venv
```

---

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 3️⃣ Run the application

```
python -m src.app
```

---

# ❗ Why `python -m src.app` is used

It may seem more natural to run:

```
python src/app.py
```

However, doing this can produce the error:

```
ModuleNotFoundError: No module named 'src'
```

This happens because Python does not treat `src` as a package when executed this way.

Running:

```
python -m src.app
```

tells Python:

> Execute the module `app` inside the `src` package.

This ensures that imports work correctly, such as:

```
from src.infer import load_model
from src.data_prep import preprocess
```

---

# 🧪 Project Status

The project is currently in an **early development stage**, focusing on:

* Dataset preparation
* Preprocessing pipeline
* Base project structure

Future improvements may include:

* Training and optimizing the CNN model
* Model evaluation
* Improved web interface
* Application deployment

---

# 👨‍💻 Author

Developed by **Bashaar Ayyoub**

