"""
Script untuk melatih ulang model Stacking Regressor.
Jalankan: python train_model.py
"""

import pandas as pd
import numpy as np
import json
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor

# ── Load data
df = pd.read_csv("laptop_rupiah.csv")

cat_cols = ["cpu", "gpu", "tipe_panel", "sistem_operasi", "fitur_keamanan"]
num_cols = ["tahun_keluaran", "ram_gb", "storage_gb", "ukuran_layar_inch",
            "resolusi_lebar", "resolusi_tinggi", "kapasitas_baterai_wh"]

X = df[cat_cols + num_cols]
y = df["harga_idr"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Preprocessing
preprocessor = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ("num", "passthrough", num_cols),
])

# ── Stacking Regressor
base_estimators = [
    ("rf",  RandomForestRegressor(n_estimators=200, random_state=42)),
    ("gb",  GradientBoostingRegressor(n_estimators=200, random_state=42)),
    ("xgb", XGBRegressor(n_estimators=200, random_state=42, verbosity=0)),
]
stack = StackingRegressor(estimators=base_estimators, final_estimator=Ridge(), cv=5)

# ── Pipeline
pipe = Pipeline([("prep", preprocessor), ("model", stack)])
print("Training model...")
pipe.fit(X_train, y_train)

# ── Evaluasi
y_pred = pipe.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n=== Hasil Evaluasi ===")
print(f"MAE : Rp {mae:,.0f}")
print(f"R²  : {r2:.4f}")
print(f"RMSE: Rp {rmse:,.0f}")

# ── Simpan model
joblib.dump(pipe, "model.joblib")

# ── Simpan metadata
meta = {
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "mae": int(mae),
    "r2": round(r2, 4),
    "rmse": int(rmse),
    "cpu_vals": sorted(df["cpu"].unique().tolist()),
    "gpu_vals": sorted(df["gpu"].unique().tolist()),
    "tipe_panel_vals": sorted(df["tipe_panel"].unique().tolist()),
    "sistem_operasi_vals": sorted(df["sistem_operasi"].unique().tolist()),
    "fitur_keamanan_vals": sorted(df["fitur_keamanan"].unique().tolist()),
    "ram_vals": sorted(df["ram_gb"].unique().tolist()),
    "storage_vals": sorted(df["storage_gb"].unique().tolist()),
    "tahun_vals": sorted(df["tahun_keluaran"].unique().tolist()),
}
with open("model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n✅ Model dan metadata tersimpan!")
