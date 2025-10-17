# src/app.py
import os
from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image

from src.infer import load_model, predict_proba, get_class_names

# Rutas base: app.py está en src/, plantillas y estáticos en la raíz
ROOT_DIR = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = ROOT_DIR / "templates"
STATIC_DIR = ROOT_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "gif"}

app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR),
)
app.config["SECRET_KEY"] = "cambia-esto-por-uno-mas-seguro"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

# Carga única del modelo
model = load_model()
CLASS_NAMES = get_class_names()

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("Sube una imagen.")
            return redirect(url_for("index"))

        if not allowed_file(file.filename):
            flash("Formato no soportado. Usa png/jpg/jpeg/bmp/gif.")
            return redirect(url_for("index"))

        # Guardar upload con nombre seguro
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        base = secure_filename(file.filename)
        name, ext = os.path.splitext(base)
        filename = f"{name}-{uuid4().hex[:8]}{ext.lower()}"
        save_path = UPLOAD_DIR / filename
        file.save(save_path)

        # Inferencia
        pil_img = Image.open(save_path)
        probs = predict_proba(model, pil_img)  # ndarray (10,)
        top_idx = int(probs.argmax())
        top_label = CLASS_NAMES[top_idx]
        top_prob = float(probs[top_idx])

        # Top-3 y Top-10 ordenados
        all_probs = sorted(
            [(CLASS_NAMES[i], float(probs[i])) for i in range(len(CLASS_NAMES))],
            key=lambda x: x[1], reverse=True
        )
        top3 = all_probs[:3]

        return render_template(
            "index.html",
            image_url=url_for("static", filename=f"uploads/{filename}"),
            top_label=top_label,
            top_prob=top_prob,
            top3=top3,
            all_probs=all_probs
        )

    return render_template("index.html", all_probs=None)

@app.route("/health")
def health():
    return {"status": "ok", "model_loaded": True}, 200

if __name__ == "__main__":
    # Ejecuta desde la raíz: python -m src.app
    app.run(host="0.0.0.0", port=5000, debug=True)


