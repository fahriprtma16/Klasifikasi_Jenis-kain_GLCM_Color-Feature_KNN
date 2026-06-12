from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"
MODEL_PATH = MODEL_DIR / "model_glcm_color_knn.pkl"
REPORT_PATH = RESULT_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = RESULT_DIR / "confusion_matrix.png"
FEATURE_CSV_PATH = RESULT_DIR / "fitur_glcm_color_dataset.csv"
METRICS_PATH = RESULT_DIR / "metrics_summary.txt"

# Dataset yang direkomendasikan agar peluang akurasi >70% lebih besar.
TARGET_CLASSES = ["Cotton", "Denim", "Nylon", "Silk", "Wool"]

IMAGE_SIZE = (256, 256)
GRAY_LEVELS = 64
DISTANCES = [1, 2, 3]
ANGLES = [0, 45, 90, 135]
GLCM_PROPERTIES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]

# Agar dataset antar kelas seimbang. Ubah None jika ingin memakai semua gambar.
BALANCE_CLASSES = True
MAX_IMAGES_PER_CLASS = 300
RANDOM_STATE = 42
TEST_SIZE = 0.20

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"}
