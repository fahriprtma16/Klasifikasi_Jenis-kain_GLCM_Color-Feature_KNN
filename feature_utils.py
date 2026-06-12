import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

from config import (
    ALLOWED_EXTENSIONS,
    ANGLES,
    BALANCE_CLASSES,
    DISTANCES,
    GLCM_PROPERTIES,
    GRAY_LEVELS,
    IMAGE_SIZE,
    MAX_IMAGES_PER_CLASS,
    RANDOM_STATE,
    TARGET_CLASSES,
)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def normalize_label(name: str) -> str:
    """Menyamakan nama folder agar Cotton/cotton/COTTON tetap terbaca."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def target_class_map() -> Dict[str, str]:
    return {normalize_label(label): label for label in TARGET_CLASSES}


def get_class_folders(dataset_dir: Path) -> List[Path]:
    """Mengambil hanya folder kelas yang direkomendasikan: Cotton, Denim, Nylon, Silk, Wool."""
    if not dataset_dir.exists():
        return []

    targets = target_class_map()
    folders = []
    for p in dataset_dir.iterdir():
        if not p.is_dir() or p.name.startswith("."):
            continue
        if normalize_label(p.name) in targets:
            folders.append(p)
    return sorted(folders, key=lambda x: targets.get(normalize_label(x.name), x.name))


def canonical_label(folder_name: str) -> str:
    return target_class_map().get(normalize_label(folder_name), folder_name)


def list_image_paths(class_folder: Path) -> List[Path]:
    """Membaca gambar secara recursive, cocok untuk struktur Kaggle: Cotton/65/im_1.png."""
    paths = []
    for root, _, files in os.walk(class_folder):
        for filename in files:
            if allowed_file(filename):
                paths.append(Path(root) / filename)
    return sorted(paths)


def preprocess_gray(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Gambar tidak bisa dibaca: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMAGE_SIZE)
    gray_quantized = (gray / (256 / GRAY_LEVELS)).astype(np.uint8)
    return gray_quantized


def preprocess_color(image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Gambar tidak bisa dibaca: {image_path}")

    image_bgr = cv2.resize(image_bgr, IMAGE_SIZE)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return image_rgb, image_hsv


def extract_glcm_features(image_path: Path) -> np.ndarray:
    gray = preprocess_gray(image_path)
    angles_rad = [np.deg2rad(a) for a in ANGLES]

    glcm = graycomatrix(
        gray,
        distances=DISTANCES,
        angles=angles_rad,
        levels=GRAY_LEVELS,
        symmetric=True,
        normed=True,
    )

    features = []
    for prop in GLCM_PROPERTIES:
        values = graycoprops(glcm, prop).ravel()
        features.append(float(np.mean(values)))
        features.append(float(np.std(values)))
    return np.array(features, dtype=np.float64)


def extract_color_features(image_path: Path) -> np.ndarray:
    """
    Fitur warna untuk membantu kain yang teksturnya mirip.
    Digunakan:
    - Mean dan standard deviation RGB
    - Mean dan standard deviation HSV
    - Histogram HSV sederhana: H=8 bin, S=4 bin, V=4 bin
    """
    image_rgb, image_hsv = preprocess_color(image_path)

    rgb_mean = image_rgb.reshape(-1, 3).mean(axis=0)
    rgb_std = image_rgb.reshape(-1, 3).std(axis=0)

    hsv_flat = image_hsv.reshape(-1, 3)
    hsv_mean = hsv_flat.mean(axis=0)
    hsv_std = hsv_flat.std(axis=0)

    hist_h = cv2.calcHist([image_hsv], [0], None, [8], [0, 180]).flatten()
    hist_s = cv2.calcHist([image_hsv], [1], None, [4], [0, 256]).flatten()
    hist_v = cv2.calcHist([image_hsv], [2], None, [4], [0, 256]).flatten()
    hist = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float64)
    hist = hist / (hist.sum() + 1e-8)

    return np.concatenate([rgb_mean, rgb_std, hsv_mean, hsv_std, hist]).astype(np.float64)


def extract_features(image_path: str | Path) -> np.ndarray:
    image_path = Path(image_path)
    glcm_features = extract_glcm_features(image_path)
    color_features = extract_color_features(image_path)
    return np.concatenate([glcm_features, color_features]).astype(np.float64)


def get_feature_names() -> List[str]:
    names = []
    for prop in GLCM_PROPERTIES:
        names.append(f"glcm_{prop}_mean")
        names.append(f"glcm_{prop}_std")

    names += [
        "rgb_mean_r", "rgb_mean_g", "rgb_mean_b",
        "rgb_std_r", "rgb_std_g", "rgb_std_b",
        "hsv_mean_h", "hsv_mean_s", "hsv_mean_v",
        "hsv_std_h", "hsv_std_s", "hsv_std_v",
    ]
    names += [f"hist_h_{i}" for i in range(8)]
    names += [f"hist_s_{i}" for i in range(4)]
    names += [f"hist_v_{i}" for i in range(4)]
    return names


def collect_dataset_paths(dataset_dir: Path):
    class_folders = get_class_folders(dataset_dir)
    data = []
    rng = random.Random(RANDOM_STATE)

    for class_folder in class_folders:
        label = canonical_label(class_folder.name)
        paths = list_image_paths(class_folder)
        rng.shuffle(paths)

        if MAX_IMAGES_PER_CLASS is not None:
            paths = paths[:MAX_IMAGES_PER_CLASS]

        for path in paths:
            data.append((path, label))

    if BALANCE_CLASSES and data:
        by_class = {}
        for path, label in data:
            by_class.setdefault(label, []).append(path)

        min_count = min(len(paths) for paths in by_class.values())
        balanced = []
        for label, paths in by_class.items():
            rng.shuffle(paths)
            for path in paths[:min_count]:
                balanced.append((path, label))
        rng.shuffle(balanced)
        return balanced

    rng.shuffle(data)
    return data


def load_dataset_features(dataset_dir: Path):
    X, y, image_paths = [], [], []
    data = collect_dataset_paths(dataset_dir)

    for index, (path, label) in enumerate(data, start=1):
        try:
            features = extract_features(path)
            X.append(features)
            y.append(label)
            image_paths.append(str(path))
            print(f"[{index}/{len(data)}] ✓ {label}: {path.name}")
        except Exception as exc:
            print(f"[{index}/{len(data)}] ✗ Gagal memproses {path}: {exc}")

    return np.array(X, dtype=np.float64), np.array(y), image_paths
