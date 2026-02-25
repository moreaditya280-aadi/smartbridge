import os
import logging
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash
import numpy as np
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = "greenclassify-secret-key"  # required for flash()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH    = os.path.join(BASE_DIR, "models", "vegetable_model.h5")
TRAIN_DIR     = os.path.join(BASE_DIR, "code", "Vegetable Images", "train")

IMG_HEIGHT = 128
IMG_WIDTH  = 128

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the uploads directory always exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# Class names  (filter out hidden files such as .DS_Store)
# ---------------------------------------------------------------------------
def load_class_names(train_dir: str) -> list[str]:
    if not os.path.isdir(train_dir):
        logging.warning("Training directory not found: %s", train_dir)
        return []
    return sorted(
        entry for entry in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, entry)) and not entry.startswith(".")
    )

class_names: list[str] = load_class_names(TRAIN_DIR)

# ---------------------------------------------------------------------------
# Model loading (lazy / guarded so startup errors are human-readable)
# ---------------------------------------------------------------------------
model = None

def get_model():
    """Load model on first call; return None if unavailable."""
    global model
    if model is not None:
        return model
    if not os.path.isfile(MODEL_PATH):
        logging.error(
            "Model file not found: %s\n"
            "Run  python train.py  first to train and save the model.",
            MODEL_PATH,
        )
        return None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        logging.info("Model loaded successfully from %s", MODEL_PATH)
        return model
    except Exception as exc:
        logging.error("Failed to load model: %s", exc)
        return None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/prediction")
def prediction():
    return render_template("prediction.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/predict", methods=["POST"])
def predict():
    # --- Validate file presence ---
    if "image" not in request.files or request.files["image"].filename == "":
        flash("No image selected. Please choose an image file.")
        return redirect(url_for("prediction"))

    file = request.files["image"]

    # --- Validate extension ---
    if not allowed_file(file.filename):
        flash(f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
        return redirect(url_for("prediction"))

    # --- Save file safely ---
    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(image_path)

    # --- Load model ---
    mdl = get_model()
    if mdl is None:
        flash("Model not loaded. Please run  python train.py  first.")
        return redirect(url_for("prediction"))

    if not class_names:
        flash("Class labels could not be loaded. Check the training data directory.")
        return redirect(url_for("prediction"))

    # --- Preprocess & predict ---
    try:
        from tensorflow.keras.preprocessing.image import load_img, img_to_array

        img       = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions  = mdl.predict(img_array)
        class_index  = int(np.argmax(predictions))
        confidence   = float(np.max(predictions)) * 100
        label        = class_names[class_index]

    except Exception as exc:
        logging.error("Prediction error: %s", exc)
        flash(f"Could not process image: {exc}")
        return redirect(url_for("prediction"))

    return render_template(
        "logout.html",
        label=label,
        confidence=round(confidence, 2),
        image_file=filename,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Pre-load model at startup so the first request isn't slow
    get_model()

    print("\n[OK]  GreenClassify is running - open http://127.0.0.1:5000/\n")
    app.run(debug=True, use_reloader=False)
