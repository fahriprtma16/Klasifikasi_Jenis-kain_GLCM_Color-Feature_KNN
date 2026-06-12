from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    CONFUSION_MATRIX_PATH,
    DATASET_DIR,
    FEATURE_CSV_PATH,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    REPORT_PATH,
    RESULT_DIR,
    TARGET_CLASSES,
    TEST_SIZE,
)
from feature_utils import get_class_folders, get_feature_names, load_dataset_features

ALGORITHM = "knn"
ALGORITHM_LABEL = "GLCM + Color + KNN"
PROJECT_TITLE = "Klasifikasi Jenis Kain 5 Kelas Menggunakan Fitur GLCM dan Color Feature dengan Algoritma KNN"


def build_model_and_grid():
    if ALGORITHM == "knn":
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier()),
            ]
        )
        param_grid = {
            "knn__n_neighbors": [3, 5, 7],
            "knn__weights": ["distance"],
            "knn__metric": ["manhattan", "euclidean"],
        }
        return pipeline, param_grid

    if ALGORITHM == "random_forest":
        pipeline = Pipeline(
            steps=[
                ("rf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        )
        param_grid = {
            "rf__n_estimators": [200, 400],
            "rf__max_depth": [None, 20, 30],
            "rf__min_samples_split": [2, 5],
            "rf__max_features": ["sqrt", "log2"],
            "rf__class_weight": ["balanced", "balanced_subsample"],
        }
        return pipeline, param_grid

    raise ValueError(f"Algoritma tidak dikenal: {ALGORITHM}")


def train_and_evaluate():
    MODEL_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    class_folders = get_class_folders(DATASET_DIR)
    detected_classes = [p.name for p in class_folders]
    if len(class_folders) < 2:
        raise RuntimeError(
            "Dataset minimal harus berisi 2 folder dari kelas target: "
            + ", ".join(TARGET_CLASSES)
        )

    print("=" * 80)
    print(PROJECT_TITLE)
    print("=" * 80)
    print(f"Folder dataset: {DATASET_DIR}")
    print(f"Kelas target: {TARGET_CLASSES}")
    print(f"Kelas terdeteksi: {detected_classes}")
    print("\nMulai ekstraksi fitur GLCM + Color Feature...")

    X, y, paths = load_dataset_features(DATASET_DIR)
    if len(X) == 0:
        raise RuntimeError("Tidak ada gambar yang berhasil diproses. Cek format dan isi folder dataset.")

    class_counts = Counter(y)
    print("\nJumlah data per kelas yang dipakai:")
    for label, total in sorted(class_counts.items()):
        print(f"- {label}: {total} gambar")

    if len(class_counts) < 2:
        raise RuntimeError("Minimal harus ada 2 kelas yang valid setelah filter TARGET_CLASSES.")

    feature_names = get_feature_names()
    df_features = pd.DataFrame(X, columns=feature_names)
    df_features.insert(0, "label", y)
    df_features.insert(0, "image_path", paths)
    df_features.to_csv(FEATURE_CSV_PATH, index=False)
    print(f"\nCSV fitur disimpan: {FEATURE_CSV_PATH}")

    can_stratify = min(class_counts.values()) >= 2
    stratify = y if can_stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
        stratify=y,
    )

    min_train_count = min(Counter(y_train).values())
    cv_splits = min(5, min_train_count)

    base_model, param_grid = build_model_and_grid()
    if cv_splits >= 2:
        print(f"\nMelakukan GridSearchCV {ALGORITHM_LABEL} dengan cv={cv_splits}...")
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
        best_cv_score = float(search.best_score_)
    else:
        print("\nDataset terlalu kecil untuk GridSearchCV. Model dilatih dengan parameter default.")
        model = base_model
        model.fit(X_train, y_train)
        best_params = {}
        best_cv_score = None

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)

    print("\nHasil evaluasi:")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall   : {macro_recall:.4f}")
    print(f"Macro F1-score : {macro_f1:.4f}")
    print(f"Best Params    : {best_params}")
    if best_cv_score is not None:
        print(f"Best CV F1     : {best_cv_score:.4f}")
    print(report)

    labels = sorted(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(9, 7))
    disp.plot(ax=ax, xticks_rotation=45, values_format="d")
    plt.title(f"Confusion Matrix - {ALGORITHM_LABEL}")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=160)
    plt.close()
    print(f"Confusion matrix disimpan: {CONFUSION_MATRIX_PATH}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(PROJECT_TITLE + "\n")
        f.write("=" * 80 + "\n")
        f.write(f"Metode: {ALGORITHM_LABEL}\n")
        f.write(f"Kelas target: {TARGET_CLASSES}\n")
        f.write(f"Total data: {len(X)}\n")
        f.write(f"Data train: {len(X_train)}\n")
        f.write(f"Data test: {len(X_test)}\n")
        f.write(f"Jumlah data per kelas: {dict(sorted(class_counts.items()))}\n")
        f.write(f"Best params: {best_params}\n")
        if best_cv_score is not None:
            f.write(f"Best CV F1: {best_cv_score:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro Precision: {macro_precision:.4f}\n")
        f.write(f"Macro Recall: {macro_recall:.4f}\n")
        f.write(f"Macro F1-score: {macro_f1:.4f}\n\n")
        f.write(report)
    print(f"Report disimpan: {REPORT_PATH}")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Accuracy={acc:.4f}\n")
        f.write(f"Macro Precision={macro_precision:.4f}\n")
        f.write(f"Macro Recall={macro_recall:.4f}\n")
        f.write(f"Macro F1-score={macro_f1:.4f}\n")
        f.write(f"Best Params={best_params}\n")

    model_bundle = {
        "model": model,
        "algorithm": ALGORITHM_LABEL,
        "feature_names": feature_names,
        "classes": list(model.classes_),
        "target_classes": TARGET_CLASSES,
        "accuracy": float(acc),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "best_params": best_params,
        "best_cv_score": best_cv_score,
    }
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"Model disimpan: {MODEL_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    train_and_evaluate()
