# 🖥️ Prediksi Harga Laptop — Machine Learning App

Aplikasi prediksi harga laptop berbasis **Stacking Regressor** yang dibangun dengan Python & Streamlit.  
Dibuat untuk keperluan akademis — Universitas AMIKOM Yogyakarta.

---

## 🎯 Performa Model

| Metrik | Nilai |
|--------|-------|
| R² Score | 0.9551 |
| MAE | Rp 1.444.649 |
| RMSE | Rp 2.087.656 |

---

## 🏗️ Arsitektur Model

```
StackingRegressor
├── Base Estimators (CV=5)
│   ├── Random Forest     (n_estimators=200)
│   ├── Gradient Boosting (n_estimators=200)
│   └── XGBoost           (n_estimators=200)
└── Final Estimator
    └── Ridge Regression
```

---

## 📁 Struktur Proyek

```
laptop_price_predictor/
├── app.py                  ← Aplikasi Streamlit utama
├── model.joblib            ← Model terlatih (Stacking Regressor)
├── model_meta.json         ← Metadata model & nilai unik fitur
├── laptop_rupiah.csv       ← Dataset (1000 baris, 14 kolom)
├── requirements.txt        ← Dependensi Python
├── runtime.txt             ← Versi Python (3.11)
└── README.md
```

---

## 🚀 Deploy ke Streamlit Cloud

### Step 1 — Upload ke GitHub
```bash
git init
git add .
git commit -m "feat: laptop price prediction ML app"
git remote add origin https://github.com/USERNAME/laptop-price-predictor.git
git push -u origin main
```

### Step 2 — Deploy di Streamlit Cloud
1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Klik **New app**
3. Pilih repository GitHub kamu
4. Set **Main file path** → `app.py`
5. Klik **Deploy!** 🚀

---

## 💻 Jalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Fitur Aplikasi

| Halaman | Konten |
|---------|--------|
| 🏠 Beranda | Overview, statistik dataset, preview data |
| 🔮 Prediksi Harga | Input spesifikasi → output harga + rentang estimasi |
| 📊 Visualisasi & EDA | Distribusi harga, fitur vs harga, korelasi, perbandingan merek |
| 📈 Performa Model | Scatter plot aktual vs prediksi, distribusi residual, metrik evaluasi |

---

## 🛠️ Tech Stack

- **Python 3.11**
- **scikit-learn** — Pipeline, StackingRegressor, preprocessing
- **XGBoost** — Gradient boosted trees
- **Streamlit** — Web app framework
- **Matplotlib / Seaborn** — Visualisasi
- **Pandas / NumPy** — Data processing
