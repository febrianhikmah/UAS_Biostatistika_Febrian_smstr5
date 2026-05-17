# Klasifikasi Penyakit Batu Empedu Menggunakan Regresi Logistik Biner

**Live App:** https://uasbiostatistikafebriansmstr-5.streamlit.app/

Project ini merupakan aplikasi analisis data berbasis Streamlit untuk klasifikasi penyakit batu empedu menggunakan metode regresi logistik biner. Aplikasi ini dibuat sebagai project Data Analysis/Biostatistika dengan alur analisis mulai dari pemahaman dataset, preprocessing, exploratory data analysis, inferensi statistika, machine learning, hingga aplikasi prediksi risiko pasien.

## Identitas Project

- Nama: Febrian Hikmah Nur Rohim
- NIM: B2D023016
- Program Studi: S1 Sains Data
- Universitas: Universitas Muhammadiyah Semarang
- Topik: Klasifikasi Penyakit Batu Empedu
- Metode utama: Regresi Logistik Biner

## Dataset

Dataset yang digunakan adalah dataset Gallstone dari UCI Machine Learning Repository.

Sumber dataset:
https://archive.ics.uci.edu/dataset/1150/gallstone-1

Dataset berisi data klinis pasien, termasuk faktor demografis, kondisi klinis, komposisi tubuh, dan indikator laboratorium. Pada project ini, variabel target yang digunakan adalah `Y_Gallstone Status`, sedangkan variabel prediktor terdiri dari beberapa fitur seperti usia, jenis kelamin, diabetes mellitus, BMI, visceral fat area, glucose, cholesterol, triglyceride, ALP, dan vitamin D.

## Fitur Aplikasi

Aplikasi Streamlit ini memiliki beberapa menu utama:

1. About Dataset
2. Tinjauan Pustaka
3. Preprocessing
4. Dashboard/EDA
5. Inferensi Statistika
6. Machine Learning
7. Prediction App
8. Contact Me

## Alur Analisis

Project ini mencakup tahapan berikut:

1. Memuat dan menjelaskan dataset.
2. Melakukan pengecekan data tidak seimbang, missing values, outlier, skala data, dan korelasi antar variabel.
3. Melakukan exploratory data analysis menggunakan visualisasi univariat, bivariat, kategorik, dan multivariat.
4. Melakukan inferensi statistika regresi logistik, meliputi estimasi parameter, Likelihood Ratio Test, Wald Test, odds ratio, dan interpretasi koefisien.
5. Melatih model Logistic Regression.
6. Mengevaluasi model menggunakan confusion matrix, accuracy, precision, recall, F1-score, dan ROC-AUC.
7. Menyediakan aplikasi prediksi risiko batu empedu berdasarkan input data pasien.

## Struktur File

```text
.
|-- utama.py                                          # File utama aplikasi Streamlit
|-- tentang_data.py                                   # Halaman penjelasan dataset
|-- pembelajaran_model.py                             # Tinjauan pustaka regresi logistik
|-- Preprocessing.py                                  # Tahapan preprocessing data
|-- visulisasi_uas.py                                 # Dashboard dan EDA
|-- Inferensi_Statistika.py                           # Inferensi statistika regresi logistik
|-- ML_uas.py                                         # Training dan evaluasi model ML
|-- prediksi.py                                       # Aplikasi prediksi risiko pasien
|-- gua.py                                            # Halaman kontak
|-- data kesehatan2.xlsx                              # Dataset awal
|-- data_kesehatan2_preprocessed.xlsx                 # Dataset hasil preprocessing
|-- logistic_regression_gallstone.pkl                 # Model tanpa preprocessing
|-- logistic_regression_gallstone_yes_proicecing.pkl  # Model dengan preprocessing
`-- requirements.txt                                  # Daftar dependensi Python
```

## Instalasi

Pastikan Python sudah terpasang, lalu install dependensi dengan perintah:

```bash
pip install -r requirements.txt
```

## Cara Menjalankan Aplikasi

Jalankan aplikasi Streamlit dari terminal:

```bash
streamlit run utama.py
```

Setelah itu, Streamlit akan menampilkan URL lokal yang dapat dibuka di browser.

## Dependensi Utama

Project ini menggunakan beberapa library utama:

- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- SciPy
- Matplotlib
- Seaborn
- Altair
- OpenPyXL

## Model

Model yang digunakan adalah Logistic Regression. Project menyediakan dua skenario analisis:

- Model menggunakan data tanpa preprocessing.
- Model menggunakan data hasil preprocessing.

Pada skenario preprocessing, normalisasi data dilakukan menggunakan Min-Max Scaling. Scaler dilatih pada data training, kemudian digunakan untuk mentransformasi data testing dan input prediksi.

## Evaluasi Model

Evaluasi model dilakukan menggunakan beberapa metrik klasifikasi:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

Metrik tersebut digunakan untuk menilai kemampuan model dalam membedakan pasien dengan dan tanpa risiko penyakit batu empedu.

## Catatan

Aplikasi prediksi pada project ini bersifat edukatif dan digunakan untuk kebutuhan analisis data. Hasil prediksi tidak dimaksudkan sebagai diagnosis medis dan tidak menggantikan konsultasi dengan tenaga kesehatan profesional.
