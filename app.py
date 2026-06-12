from pathlib import Path

import joblib
import numpy as np
from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from config import MODEL_PATH, UPLOAD_DIR
from feature_utils import allowed_file, extract_features, get_feature_names
from train_evaluate import ALGORITHM_LABEL, PROJECT_TITLE, train_and_evaluate

app = Flask(__name__)
app.secret_key = "fabric-glcm-color-classification-secret-key"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

model_bundle = None


def load_or_train_model():
    global model_bundle
    if MODEL_PATH.exists():
        model_bundle = joblib.load(MODEL_PATH)
        print(f"Model dimuat dari {MODEL_PATH}")
    else:
        print("Model belum ada. Mencoba training dari dataset...")
        train_and_evaluate()
        model_bundle = joblib.load(MODEL_PATH)


def predict_image(image_path: str | Path):
    features = extract_features(image_path)
    model = model_bundle["model"]
    prediction = model.predict(features.reshape(1, -1))[0]

    top_predictions = []
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features.reshape(1, -1))[0]
        classes = model.classes_
        sorted_idx = np.argsort(probabilities)[::-1]
        for idx in sorted_idx[:5]:
            top_predictions.append(
                {"label": classes[idx], "probability": round(float(probabilities[idx]) * 100, 2)}
            )

    feature_data = []
    names = model_bundle.get("feature_names", get_feature_names())
    for name, value in zip(names, features):
        feature_data.append({"name": name, "value": round(float(value), 6)})

    return prediction, top_predictions, feature_data


@app.route("/")
def index():
    classes = model_bundle.get("classes", []) if model_bundle else []
    accuracy = model_bundle.get("accuracy", None) if model_bundle else None
    macro_precision = model_bundle.get("macro_precision", None) if model_bundle else None
    macro_recall = model_bundle.get("macro_recall", None) if model_bundle else None
    macro_f1 = model_bundle.get("macro_f1", None) if model_bundle else None
    best_params = model_bundle.get("best_params", {}) if model_bundle else {}
    return render_template(
        "index.html",
        title=PROJECT_TITLE,
        algorithm=ALGORITHM_LABEL,
        classes=classes,
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        best_params=best_params,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model_bundle is None:
        flash("Model belum tersedia. Jalankan python train_evaluate.py terlebih dahulu.")
        return redirect(url_for("index"))

    if "file" not in request.files:
        flash("File belum dipilih.")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("File belum dipilih.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Format file tidak didukung. Gunakan jpg, jpeg, png, bmp, webp, tif, atau tiff.")
        return redirect(url_for("index"))

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filename = secure_filename(file.filename)
    save_path = UPLOAD_DIR / filename
    file.save(save_path)

    try:
        prediction, top_predictions, feature_data = predict_image(save_path)
    except Exception as exc:
        flash(f"Gagal memproses gambar: {exc}")
        return redirect(url_for("index"))

    return render_template(
        "result.html",
        title=PROJECT_TITLE,
        algorithm=ALGORITHM_LABEL,
        filename=filename,
        prediction=prediction,
        top_predictions=top_predictions,
        feature_data=feature_data[:24],
        total_features=len(feature_data),
    )


@app.route("/info")
def info():
    return render_template("info.html", title=PROJECT_TITLE, algorithm=ALGORITHM_LABEL)


if __name__ == "__main__":
    print("=" * 80)
    print(PROJECT_TITLE)
    print("=" * 80)
    try:
        load_or_train_model()
    except Exception as exc:
        print(f"PERINGATAN: Model belum bisa dilatih/dimuat: {exc}")
        print("Pastikan dataset sudah diletakkan di folder dataset/Cotton, Denim, Nylon, Silk, Wool")
    app.run(debug=True)
