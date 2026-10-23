# 💳 End-to-End Credit Risk Prediction Model
**Project-Based Internship Data Scientist — ID/X Partners x Rakamin Academy**

![R](https://img.shields.io/badge/Language-R-276DC3?style=for-the-badge&logo=r&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

---

## 📌 1. Project Overview

Proyek ini bertujuan untuk membangun model *machine learning* prediktif guna menilai risiko kredit (*credit risk*) nasabah multifinance[cite: 69, 70]. Model ini dirancang untuk mengidentifikasi probabilitas gagal bayar (*default rate*), menekan angka kredit macet (*Non-Performing Loan*), serta mendukung penerapan *risk-based pricing*[cite: 70].

* **Total Data Awal:** 466.285 baris dan 74 kolom[cite: 71].
* **Data Bersih Siap Latih:** 234.918 baris setelah pembersihan data dan eliminasi data *leakage*[cite: 72, 75].

---

## 🔍 2. Tahapan Data Engineering & Preprocessing

1. **Eliminasi Kolom Kosong & Redundan:** 13 kolom kosong 100% serta variabel dengan tingkat *missing rate* di atas 70% (seperti `mths_since_last_record`) dibuang[cite: 71].
2. **Pencegahan Data Leakage:** 19 variabel *post-origination* (seperti `total_pymnt`, `recoveries`, `last_pymnt_amnt`) dieliminasi karena informasinya baru terbentuk setelah kredit berjalan[cite: 71, 75].
3. **Filtrasi Target Variabel:** Hanya mempertahankan status pinjaman final[cite: 72]:
   * **Good Loan (0):** `Fully Paid` dan sejenisnya (184.716 data / 78,63%)[cite: 72].
   * **Bad Loan (1):** `Charged Off`, `Default`, dan pinjaman terlambat bermasalah (50.202 data / 21,37%)[cite: 72].
4. **Outlier Handling & Imputasi:** Penerapan *Winsorizing* (persentil 1%-99%) pada fitur numerik ekstrem serta imputasi median dan pembuatan biner *missing flag* (`never_delinq`)[cite: 76].
5. **Train-Test Split:** Menggunakan *Stratified Random Sampling* dengan rasio **80:20** (187.935 data *train* dan 46.983 data *test*)[cite: 76].

---

## 🤖 3. Pemodelan Machine Learning & Evaluasi

Dua algoritma utama dibandingkan untuk mencari performa terbaik[cite: 77]:

### A. Logistic Regression (Baseline Model)
* **Konfigurasi:** Dievaluasi menggunakan GridSearchCV dengan *3-fold Stratified Cross-Validation* ($C \in \{0.01, 0.1, 1, 10\}$) dan parameter `class_weight='balanced'`[cite: 77].
* **Hasil Performa:** Nilai $C = 10$ menghasilkan CV ROC-AUC **0,7049**[cite: 77], serta ROC-AUC pada *Test Set* sebesar **0,6995** (Gap train-test sangat kecil: 0,0011, menunjukkan generalisasi yang sangat stabil)[cite: 78].

### B. Random Forest (Pembanding Utama)
* **Konfigurasi:** Menggunakan *validation split* (75:25), penyesuaian `class_weight='balanced'`, dengan hyperparameter terbaik `n_estimators=150`, `max_depth=16`, dan `min_samples_leaf=20`[cite: 77].
* **Hasil Performa:** Mencapai Validation ROC-AUC **0,7127**[cite: 77] dan ROC-AUC pada *Test Set* sebesar **0,7025**[cite: 78].

### C. Perbandingan Metrik Evaluasi
* **Random Forest** unggul di 4 dari 5 metrik utama: Akurasi (**65,1%**), Presisi (**33,36%**), F1-Score (**0,4374**), dan ROC-AUC (**0,7025**)[cite: 79].
* **Insight Bisnis:** Random Forest sangat ideal sebagai mesin *underwriting* otomatis untuk menekan risiko penolakan nasabah baik (*False Bad*), sementara Logistic Regression sangat cocok untuk kebutuhan audit regulasi karena transparansi koefisiennya[cite: 82].

---

## 📁 4. Repository Structure

```text
.
├── python/
│   ├── main_pipeline.py                  # Script pipeline utama pembersihan & pemodelan
│   └── final_submission_notebook.ipynb   # Jupyter Notebook eksplorasi & eksperimen model
├── loan_data_cleaned.csv                 # Dataset bersih siap latih
├── Final Task_IDX Partners_Data Scientist_Ahmad Jalaludin Rahman.pdf # Dokumen presentasi final task
└── README.md                             # Dokumentasi repositori
