import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ─── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Harga Laptop 🖥️",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load assets ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

@st.cache_data
def load_meta():
    with open("model_meta.json") as f:
        return json.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("laptop_rupiah.csv")

model = load_model()
meta  = load_meta()
df    = load_data()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/color/96/laptop.png", width=80
)
st.sidebar.title("🖥️ Laptop Price AI")
st.sidebar.markdown("**Universitas AMIKOM Yogyakarta**")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigasi",
    ["🏠 Beranda", "🔮 Prediksi Harga", "📊 Visualisasi & EDA", "📈 Performa Model"],
)

# Helper: format Rupiah
def fmt_idr(x):
    return f"Rp {x:,.0f}".replace(",", ".")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — BERANDA
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Beranda":
    st.title("🖥️ Prediksi Harga Laptop dengan Machine Learning")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 Total Data", f"{len(df):,} laptop")
    col2.metric("✅ R² Score", f"{meta['r2']:.4f}")
    col3.metric("📉 MAE", fmt_idr(meta['mae']))
    col4.metric("📐 RMSE", fmt_idr(meta['rmse']))

    st.markdown("---")
    st.subheader("Tentang Proyek")
    st.markdown("""
    Aplikasi ini memprediksi **harga laptop dalam Rupiah (IDR)** menggunakan model 
    **Stacking Regressor** yang menggabungkan tiga algoritma ensemble:

    | Model | Peran |
    |---|---|
    | 🌲 Random Forest | Base estimator |
    | 📈 Gradient Boosting | Base estimator |
    | ⚡ XGBoost | Base estimator |
    | 🎯 Ridge Regression | Meta-learner (final estimator) |

    **Fitur yang digunakan:**
    - Spesifikasi teknis: CPU, GPU, RAM, Storage, Ukuran Layar, Resolusi, Panel, Baterai
    - Ekosistem: Sistem Operasi, Fitur Keamanan
    - Informasi rilis: Tahun Keluaran
    """)

    st.subheader("Preview Dataset")
    st.dataframe(df.head(10), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDIKSI HARGA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediksi Harga":
    st.title("🔮 Prediksi Harga Laptop")
    st.markdown("Isi spesifikasi laptop di bawah ini, lalu klik **Prediksi**.")
    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("⚙️ Spesifikasi Utama")
        tahun   = st.selectbox("Tahun Keluaran", sorted(meta["tahun_vals"], reverse=True))
        cpu     = st.selectbox("Prosesor (CPU)", meta["cpu_vals"])
        ram     = st.selectbox("RAM (GB)", meta["ram_vals"])
        storage = st.selectbox("Storage (GB)", meta["storage_vals"])
        gpu     = st.selectbox("GPU / Graphics", meta["gpu_vals"])

    with col_b:
        st.subheader("🖥️ Layar & Sistem")
        layar   = st.slider("Ukuran Layar (inch)", 11.0, 18.0, 14.0, 0.1)
        panel   = st.selectbox("Tipe Panel", meta["tipe_panel_vals"])
        res_w   = st.selectbox("Resolusi Lebar (px)", sorted(df["resolusi_lebar"].unique().tolist()))
        res_h   = st.selectbox("Resolusi Tinggi (px)", sorted(df["resolusi_tinggi"].unique().tolist()))
        baterai = st.slider("Kapasitas Baterai (Wh)", 35, 99, 56)
        os_val  = st.selectbox("Sistem Operasi", meta["sistem_operasi_vals"])
        keamanan= st.selectbox("Fitur Keamanan", meta["fitur_keamanan_vals"])

    st.markdown("---")

    if st.button("🔮 Prediksi Harga Sekarang!", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            "cpu": cpu, "gpu": gpu, "tipe_panel": panel,
            "sistem_operasi": os_val, "fitur_keamanan": keamanan,
            "tahun_keluaran": tahun, "ram_gb": ram, "storage_gb": storage,
            "ukuran_layar_inch": layar, "resolusi_lebar": res_w,
            "resolusi_tinggi": res_h, "kapasitas_baterai_wh": baterai,
        }])
        pred = model.predict(input_df)[0]

        st.success(f"### 💰 Prediksi Harga: **{fmt_idr(pred)}**")

        # Confidence range ±MAE
        low  = max(0, pred - meta["mae"])
        high = pred + meta["mae"]
        st.info(f"📏 Estimasi rentang harga: **{fmt_idr(low)}** — **{fmt_idr(high)}**")

        # Benchmark chart
        st.markdown("#### Posisi Harga dalam Distribusi")
        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        ax.hist(df["harga_idr"], bins=40, color="#4f8bf9", alpha=0.7, edgecolor="none")
        ax.axvline(pred, color="#ff4b4b", linewidth=2.5, label=f"Prediksi: {fmt_idr(pred)}")
        ax.axvline(df["harga_idr"].median(), color="#ffa500", linewidth=1.5, linestyle="--", label="Median dataset")
        ax.set_xlabel("Harga (IDR)", color="white")
        ax.set_ylabel("Jumlah Laptop", color="white")
        ax.tick_params(colors="white")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rp {x/1e6:.0f}Jt"))
        ax.legend(facecolor="#1e1e2e", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — VISUALISASI & EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Visualisasi & EDA":
    st.title("📊 Eksplorasi Data (EDA)")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Distribusi Harga", "Fitur vs Harga", "Korelasi", "Perbandingan Merek"])

    # ── Tab 1: Distribusi harga ──
    with tab1:
        st.subheader("Distribusi Harga Laptop")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#0e1117")
        for ax in axes:
            ax.set_facecolor("#161b22")

        axes[0].hist(df["harga_idr"], bins=40, color="#4f8bf9", edgecolor="none", alpha=0.85)
        axes[0].set_title("Distribusi Harga", color="white")
        axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rp {x/1e6:.0f}Jt"))

        axes[1].boxplot(df["harga_idr"], patch_artist=True,
                        boxprops=dict(facecolor="#4f8bf9", color="#4f8bf9"),
                        medianprops=dict(color="white", linewidth=2),
                        whiskerprops=dict(color="white"),
                        capprops=dict(color="white"),
                        flierprops=dict(markerfacecolor="#ff4b4b", markersize=4))
        axes[1].set_title("Box Plot Harga", color="white")
        axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rp {x/1e6:.0f}Jt"))

        for ax in axes:
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Min", fmt_idr(df["harga_idr"].min()))
        col2.metric("Median", fmt_idr(df["harga_idr"].median()))
        col3.metric("Mean", fmt_idr(df["harga_idr"].mean()))
        col4.metric("Max", fmt_idr(df["harga_idr"].max()))

    # ── Tab 2: Fitur vs Harga ──
    with tab2:
        st.subheader("Pengaruh Fitur terhadap Harga")
        fitur = st.selectbox("Pilih fitur:", ["cpu","gpu","tipe_panel","sistem_operasi","fitur_keamanan","ram_gb","tahun_keluaran"])
        grp = df.groupby(fitur)["harga_idr"].median().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#161b22")
        bars = ax.barh(grp.index.astype(str), grp.values, color="#4f8bf9", alpha=0.85)
        ax.set_xlabel("Median Harga (IDR)", color="white")
        ax.set_title(f"Median Harga per {fitur}", color="white")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rp {x/1e6:.0f}Jt"))
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        # value labels
        for bar in bars:
            ax.text(bar.get_width() + 100000, bar.get_y() + bar.get_height()/2,
                    fmt_idr(bar.get_width()), va="center", color="white", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

    # ── Tab 3: Korelasi ──
    with tab3:
        st.subheader("Matriks Korelasi Fitur Numerik")
        num_df = df[["tahun_keluaran","ram_gb","storage_gb","ukuran_layar_inch",
                     "resolusi_lebar","resolusi_tinggi","kapasitas_baterai_wh","harga_idr"]]
        corr = num_df.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    ax=ax, linewidths=0.5, linecolor="#0e1117",
                    annot_kws={"size": 10, "color": "white"})
        ax.tick_params(colors="white")
        ax.set_title("Korelasi Fitur Numerik", color="white")
        st.pyplot(fig)

    # ── Tab 4: Perbandingan merek ──
    with tab4:
        st.subheader("Rata-rata Harga per Merek Laptop")
        df["merek"] = df["nama_laptop"].str.split().str[0]
        brand_avg = df.groupby("merek")["harga_idr"].mean().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#161b22")
        colors = ["#ff4b4b" if v == brand_avg.max() else "#4f8bf9" for v in brand_avg.values]
        ax.bar(brand_avg.index, brand_avg.values, color=colors, alpha=0.85)
        ax.set_ylabel("Rata-rata Harga (IDR)", color="white")
        ax.set_title("Rata-rata Harga per Merek", color="white")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rp {x/1e6:.0f}Jt"))
        ax.tick_params(colors="white", axis="x", rotation=30)
        ax.tick_params(colors="white", axis="y")
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PERFORMA MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Performa Model":
    st.title("📈 Evaluasi & Performa Model")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{meta['r2']:.4f}", help="Semakin mendekati 1.0 semakin baik")
    col2.metric("MAE", fmt_idr(meta['mae']), help="Mean Absolute Error")
    col3.metric("RMSE", fmt_idr(meta['rmse']), help="Root Mean Squared Error")

    st.markdown("---")
    st.subheader("Arsitektur Model: Stacking Regressor")
    st.markdown("""
    ```
    StackingRegressor
    ├── Base Estimators (CV=5)
    │   ├── Random Forest     (n_estimators=200)
    │   ├── Gradient Boosting (n_estimators=200)  
    │   └── XGBoost           (n_estimators=200)
    └── Final Estimator
        └── Ridge Regression
    ```
    """)

    st.subheader("Perbandingan Prediksi vs Aktual (Simulasi 50 Data)")
    # Re-run quick train for scatter plot (cached implicitly)
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from xgboost import XGBRegressor

    cat_cols = meta["cat_cols"]
    num_cols = meta["num_cols"]
    X = df[cat_cols + num_cols]
    y = df["harga_idr"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    sample_idx = np.random.RandomState(42).choice(len(y_test), 50, replace=False)
    y_sample = np.array(y_test)[sample_idx]
    p_sample = y_pred[sample_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0e1117")

    # Scatter
    ax = axes[0]
    ax.set_facecolor("#161b22")
    ax.scatter(y_sample, p_sample, color="#4f8bf9", alpha=0.7, s=60)
    lims = [min(y_sample.min(), p_sample.min()), max(y_sample.max(), p_sample.max())]
    ax.plot(lims, lims, "r--", lw=2, label="Perfect fit")
    ax.set_xlabel("Harga Aktual", color="white")
    ax.set_ylabel("Harga Prediksi", color="white")
    ax.set_title("Aktual vs Prediksi", color="white")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}Jt"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}Jt"))
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1e1e2e", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Residuals
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    residuals = p_sample - y_sample
    ax2.hist(residuals, bins=20, color="#ffa500", alpha=0.8, edgecolor="none")
    ax2.axvline(0, color="white", linestyle="--", lw=2)
    ax2.set_xlabel("Error (Prediksi - Aktual)", color="white")
    ax2.set_ylabel("Frekuensi", color="white")
    ax2.set_title("Distribusi Residual", color="white")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}Jt"))
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.info("""
    **Interpretasi:**
    - R² = 0.9551 → Model mampu menjelaskan **95.5%** variasi harga laptop ✅
    - MAE ≈ Rp 1.44 Juta → Rata-rata selisih prediksi cukup kecil dibanding rentang harga (Rp 3.9Jt – 50.6Jt) ✅
    - Residual mendekati distribusi normal → Model tidak bias secara sistematis ✅
    """)
