# app.py
import json
from pathlib import Path

import numpy as np
import streamlit as st
import joblib
import xgboost as xgb


# ----------------------------
# Page config + simple styling
# ----------------------------
st.set_page_config(
    page_title="Credit Risk Scoring (XGBoost)",
    page_icon="💳",
    layout="wide",
)

st.markdown(
    """
<style>
/* Make the app feel cleaner */
.block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
div[data-testid="stMetricValue"] { font-size: 1.8rem; }
.small-note { font-size: 0.92rem; opacity: 0.85; }
.card {
  padding: 1rem 1.25rem; border-radius: 14px;
  border: 1px solid rgba(49, 51, 63, 0.18);
  background: rgba(255, 255, 255, 0.04);
}
.badge {
  display: inline-block; padding: 0.25rem 0.6rem; border-radius: 999px;
  border: 1px solid rgba(49, 51, 63, 0.22);
  font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# Load artifacts
# ----------------------------
APP_DIR = Path(__file__).parent
VECTORIZER_PATH = APP_DIR / "dict_vectorizer.joblib"
MODEL_PATH = APP_DIR / "xgb_credit_risk.json"


@st.cache_resource
def load_artifacts():
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {VECTORIZER_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {MODEL_PATH}")

    dv = joblib.load(VECTORIZER_PATH)

    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))

    return dv, booster


def risk_bucket(prob_default: float) -> tuple[str, str]:
    """
    Returns (label, helper_text)
    Thresholds are configurable; these are simple & demo-friendly.
    """
    if prob_default < 0.33:
        return "LOW", "Risiko rendah — umumnya layak dipertimbangkan."
    if prob_default < 0.66:
        return "MEDIUM", "Risiko sedang — perlu verifikasi tambahan."
    return "HIGH", "Risiko tinggi — disarankan evaluasi ketat."


def clamp_non_negative(value: float) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    return max(0.0, v)


# ----------------------------
# Header
# ----------------------------
st.title("💳 Credit Risk Scoring (XGBoost)")
st.caption(
    "Aplikasi ini memprediksi **probabilitas gagal bayar (Probability of Default / PD)** "
    "menggunakan model **XGBoost** dan fitur input yang sama seperti saat training."
)

with st.expander("ℹ️ Petunjuk singkat", expanded=False):
    st.markdown(
        """
- Isi data nasabah pada panel **Input** (kiri).
- Klik **Prediksi Risiko** untuk mendapatkan **PD** dan **kategori risiko**.
- Kategori risiko menggunakan ambang sederhana: **Low < 0.33**, **Medium 0.33–0.66**, **High ≥ 0.66**.
"""
    )

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("🧾 Input Nasabah")
st.sidebar.markdown('<div class="small-note">Masukkan nilai sesuai data yang tersedia.</div>', unsafe_allow_html=True)

# Categorical options (based on examples you sent)
HOME_OPTIONS = ["owner", "parents"]
MARITAL_OPTIONS = ["single", "married"]
RECORDS_OPTIONS = ["no", "yes"]
JOB_OPTIONS = ["fixed", "freelance", "partime"]

with st.sidebar:
    st.subheader("Data Kategorikal")
    home = st.selectbox("Home", options=HOME_OPTIONS, index=0, help="Status tempat tinggal.")
    marital = st.selectbox("Marital", options=MARITAL_OPTIONS, index=1, help="Status pernikahan.")
    records = st.selectbox("Records", options=RECORDS_OPTIONS, index=0, help="Ada catatan kredit bermasalah?")
    job = st.selectbox("Job", options=JOB_OPTIONS, index=1, help="Jenis pekerjaan.")

    st.subheader("Data Numerik")
    col_a, col_b = st.columns(2)

    with col_a:
        age = st.number_input("Age", min_value=0, value=36, step=1)
        seniority = st.number_input("Seniority", min_value=0, value=5, step=1, help="Lama bekerja/masa kerja.")
        time = st.number_input("Time (months)", min_value=0, value=36, step=1, help="Tenor (bulan).")
        expenses = st.number_input("Expenses", min_value=0, value=60, step=1)

    with col_b:
        income = st.number_input("Income", min_value=0.0, value=100.0, step=1.0)
        assets = st.number_input("Assets", min_value=0.0, value=4000.0, step=100.0)
        debt = st.number_input("Debt", min_value=0.0, value=0.0, step=50.0)
        amount = st.number_input("Amount", min_value=0, value=1100, step=50, help="Jumlah pinjaman.")
        price = st.number_input("Price", min_value=0, value=1400, step=50, help="Harga barang/objek pembiayaan.")

    st.divider()
    run_pred = st.button("🔮 Prediksi Risiko", use_container_width=True)


# ----------------------------
# Main area
# ----------------------------
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown("### Ringkasan Input")
    input_dict = {
        "seniority": int(seniority),
        "home": str(home),
        "time": int(time),
        "age": int(age),
        "marital": str(marital),
        "records": str(records),
        "job": str(job),
        "expenses": int(expenses),
        "income": float(income),
        "assets": float(assets),
        "debt": float(debt),
        "amount": int(amount),
        "price": int(price),
    }

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.json(input_dict, expanded=False)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Catatan")
    st.markdown(
        "- Pastikan nilai **masuk akal** (misalnya age tidak terlalu kecil/terlalu besar).\n"
        "- Model menghasilkan **probabilitas** (0–1), bukan keputusan final.\n"
        "- Gunakan hasil ini sebagai **alat bantu** (decision support)."
    )

with right:
    st.markdown("### Hasil Prediksi")

    if run_pred:
        try:
            dv, booster = load_artifacts()

            # Optional: clean numeric inputs (ensure non-negative)
            cleaned = dict(input_dict)
            for k in ["income", "assets", "debt"]:
                cleaned[k] = clamp_non_negative(cleaned[k])

            X = dv.transform([cleaned])
            dmat = xgb.DMatrix(X)

            proba = float(booster.predict(dmat)[0])
            label, helper = risk_bucket(proba)

            # Nice display
            st.markdown('<div class="card">', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Probability of Default (PD)", f"{proba:.3f}")
            with c2:
                st.metric("Kategori Risiko", label)

            st.progress(min(max(proba, 0.0), 1.0))
            st.markdown(f"<span class='badge'>Risk: {label}</span> &nbsp; {helper}", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("🔎 Lihat penjelasan kategori risiko"):
                st.markdown(
                    """
- **LOW**: PD < 0.33  
- **MEDIUM**: 0.33 ≤ PD < 0.66  
- **HIGH**: PD ≥ 0.66  

> Ambang ini untuk kebutuhan demo dan bisa disesuaikan sesuai kebijakan bisnis.
"""
                )

        except Exception as e:
            st.error("Terjadi error saat memuat model atau melakukan prediksi.")
            st.code(str(e))

    else:
        st.info("Isi input di sebelah kiri, lalu klik **Prediksi Risiko**.")

st.divider()

# Footer
st.caption(
    "Model: XGBoost (binary:logistic) • Input: DictVectorizer • Output: Probability of Default (PD)"
)
