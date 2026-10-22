# 💳 End-to-End Credit Risk Prediction Model
**Project-Based Internship Data Scientist — ID/X Partners**

![R](https://img.shields.io/badge/Language-R-276DC3?style=for-the-badge&logo=r&logoColor=white)
![Python](https://img.shields.io/badge/Language-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-00b4d8?style=for-the-badge)

---

## 📌 Project Overview
Proyek ini bertujuan untuk membangun model prediksi risiko kredit (*credit risk analytics*) end-to-end menggunakan data historis pinjaman *multifinance*. Tujuan utama dari proyek ini adalah mengidentifikasi calon peminjam berisiko tinggi guna meminimalkan angka *default* (gagal bayar) serta membantu lini bisnis perbankan/multifinance dalam mengoptimalkan keputusan persetujuan kredit (*credit approval*).

Proyek ini menerapkan **Hybrid Data Science Approach**:
1. **R Approach (`tidyverse`, `caret`, `ggplot2`)**: Berfokus pada pembersihan data mendalam (*data cleaning*), penanganan *missing values* & *outliers*, *feature engineering*, serta analisis statistik eksploratif (EDA).
---

## 🎯 Key Metrics & Results

- **Best Performing Model:** Random Forest Classifier / Logistic Regression
- **Primary Metric:** **ROC-AUC = 0.7025**
- **Business Focus:** Memaksimalkan recall pada kelas berisiko (*bad loans*) tanpa mengorbankan kepraktisan operasional bisnis pinjaman.

---

## 🛠️ Data Preprocessing & Methodology

- **Missing Value Handling:** Imputasi median/modus yang disesuaikan dengan jenis kolom numerik dan kategorikal.
- **Outlier Treatment:** Penanganan pencilan berbasis analisis distribusi data historis.
- **Feature Engineering & Transformation:** Pembentukan variabel indikator risiko baru, pembagian *binning*, serta *one-hot encoding*.
- **Class Imbalance Handling:** Penerapan teknik re-sampling/penyesuaian bobot kelas untuk memperkuat sinyal prediksi pada kategori gagal bayar.

---

## 📁 Repository Structure

```text
.
├── R/
│   ├── 01_eda_and_data_cleaning.R        # Script R untuk EDA, visualisasi, & cleaning
│   └── 02_modelling_and_evaluation.R     # Script R untuk pemodelan statistik & evaluasi
├── python/
│   ├── main_pipeline.py                  # Script Python pipeline machine learning
│   └── final_submission_notebook.ipynb   # Jupyter Notebook eksperimen model & grafik
├── loan_data_cleaned.csv                 # Dataset terisi & siap latih
├── Final Task_ID_X Partners_DS.pptx      # Slide presentasi eksekutif & rekomendasi bisnis
└── README.md                             # Dokumentasi proyek
