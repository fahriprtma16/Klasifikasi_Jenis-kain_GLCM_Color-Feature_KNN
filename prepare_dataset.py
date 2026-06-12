"""
Script opsional untuk menyalin 5 kelas rekomendasi dari dataset lama/full Kaggle.

Contoh:
python prepare_dataset.py --source "C:\\Users\\Fakhri\\Downloads\\klasifikasi_jenis_kain_glcm_knn\\klasifikasi_jenis_kain_glcm_knn\\dataset"

Script ini akan membuat struktur:
dataset/Cotton
dataset/Denim
dataset/Nylon
dataset/Silk
dataset/Wool
"""
import argparse
import shutil
from pathlib import Path

from config import DATASET_DIR, TARGET_CLASSES, ALLOWED_EXTENSIONS, MAX_IMAGES_PER_CLASS
from feature_utils import normalize_label


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower().replace('.', '') in ALLOWED_EXTENSIONS


def copy_class(source_class_dir: Path, dest_class_dir: Path, limit: int | None = None):
    dest_class_dir.mkdir(parents=True, exist_ok=True)
    images = sorted([p for p in source_class_dir.rglob('*') if is_image(p)])
    if limit is not None:
        images = images[:limit]
    for i, src in enumerate(images, start=1):
        ext = src.suffix.lower()
        dst = dest_class_dir / f"{source_class_dir.name}_{i:05d}{ext}"
        shutil.copy2(src, dst)
    return len(images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='Folder dataset lama/full Kaggle')
    parser.add_argument('--limit', type=int, default=MAX_IMAGES_PER_CLASS, help='Maksimal gambar per kelas')
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source tidak ditemukan: {source}")

    source_folders = {normalize_label(p.name): p for p in source.iterdir() if p.is_dir()}
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print('Menyalin dataset rekomendasi...')
    for label in TARGET_CLASSES:
        key = normalize_label(label)
        if key not in source_folders:
            print(f"PERINGATAN: Folder {label} tidak ditemukan di source")
            continue
        total = copy_class(source_folders[key], DATASET_DIR / label, args.limit)
        print(f"✓ {label}: {total} gambar disalin")

    print(f"Selesai. Dataset tersimpan di: {DATASET_DIR}")


if __name__ == '__main__':
    main()
