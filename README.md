# Klasifikasi Jenis Kain 5 Kelas Menggunakan Fitur GLCM dan Color Feature dengan Algoritma KNN

Project ini adalah versi optimasi dari klasifikasi jenis kain berbasis citra digital.
Metode yang digunakan: **GLCM + Color + KNN**.

## Dataset yang Direkomendasikan

Gunakan 5 kelas berikut dari dataset Kaggle The Fabrics Dataset by iBUG:

- Cotton
- Denim
- Nylon
- Silk
- Wool

Alasan pemilihan 5 kelas ini adalah karena teksturnya relatif lebih berbeda dibandingkan kelas Blended atau Polyester, sehingga peluang akurasi meningkat.

## Fitur yang Digunakan

### 1. GLCM Texture Feature
- Contrast
- Dissimilarity
- Homogeneity
- Energy
- Correlation
- ASM

Masing-masing fitur dihitung menggunakan beberapa jarak dan sudut, lalu diringkas dengan mean dan standard deviation.

### 2. Color Feature
- Mean RGB
- Standard deviation RGB
- Mean HSV
- Standard deviation HSV
- Histogram HSV sederhana

## Cara Menjalankan

### 1. Install Library

```bash
python -m pip install -r requirements.txt
```

### 2. Masukkan Dataset

Struktur dataset:

```text
dataset/
├── Cotton/
├── Denim/
├── Nylon/
├── Silk/
└── Wool/
```


### 3. Training dan Evaluasi

```bash
python train_evaluate.py
```

Output yang dihasilkan:

```text
models/model_glcm_color_knn.pkl
static/results/classification_report.txt
static/results/confusion_matrix.png
static/results/fitur_glcm_color_dataset.csv
static/results/metrics_summary.txt
```

### 4. Jalankan Web App

```bash
python app.py
```

Buka browser:

```text
http://127.0.0.1:5000
```

## Catatan Akurasi

Target akurasi bersifat estimasi. Hasil sebenarnya bergantung pada jumlah gambar, kualitas gambar, keseimbangan dataset, serta hasil pembagian train-test.
Untuk meningkatkan peluang akurasi di atas 70%, pastikan jumlah gambar tiap kelas seimbang.
