import streamlit as st
import pandas as pd
import pickle
import numpy as np

# === CONFIGURASI DASAR ===
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# === JUDUL & DESKRIPSI ===
st.title("❤️ Prediksi Risiko Penyakit Jantung")
st.write("""
Aplikasi ini memprediksi apakah seseorang **berisiko terkena penyakit jantung** berdasarkan data kesehatan dasar mereka.  
Masukkan data di bawah ini, lalu tekan **Prediksi Sekarang**.
""")

# === LOAD MODEL TERBAIK ===
with open("random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

# === INPUT FORM ===
st.subheader("🩺 Masukkan Data Anda")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Umur", min_value=20, max_value=100, value=40)
    sex = st.selectbox("Jenis Kelamin", ["Pria", "Wanita"])
    chest_pain = st.selectbox("Tipe Nyeri Dada (1-4)", [1, 2, 3, 4])
    resting_bp = st.number_input("Tekanan Darah Istirahat (mm Hg)", min_value=80, max_value=200, value=130)
    cholesterol = st.number_input("Kadar Kolesterol (mg/dl)", min_value=85, max_value=600, value=230)

with col2:
    fasting_blood_sugar = st.selectbox("Gula Darah Puasa > 120 mg/dl", ["Tidak", "Ya"])
    resting_ecg = st.selectbox("Hasil ECG", [0, 1, 2])
    max_heart_rate = st.number_input("Denyut Jantung Maksimum", min_value=60, max_value=220, value=150)
    exercise_angina = st.selectbox("Angina Saat Olahraga", ["Tidak", "Ya"])
    oldpeak = st.number_input("Oldpeak (Depresi ST)", min_value=-3.0, max_value=6.0, value=1.0)
    st_slope = st.selectbox("ST Slope (0-2)", [0, 1, 2])

# === KONVERSI INPUT KE FORMAT MODEL ===
input_data = pd.DataFrame({
    "age": [age],
    "sex": [1 if sex == "Pria" else 0],
    "chest pain type": [chest_pain],
    "resting bp s": [resting_bp],
    "cholesterol": [cholesterol],
    "fasting blood sugar": [1 if fasting_blood_sugar == "Ya" else 0],
    "resting ecg": [resting_ecg],
    "max heart rate": [max_heart_rate],
    "exercise angina": [1 if exercise_angina == "Ya" else 0],
    "oldpeak": [oldpeak],
    "ST slope": [st_slope],
    "max_heart_rate_per_age": [max_heart_rate / age]
})

# === TOMBOL PREDIKSI ===
if st.button("🚀 Prediksi Sekarang"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]  # Probabilitas terkena penyakit jantung

    st.subheader("📊 Hasil Prediksi")
    if pred == 1:
        st.error(f"⚠️ Hasil menunjukkan **RISIKO TINGGI** terkena penyakit jantung.\n\nProbabilitas: {prob:.2%}")
        st.write("Segera konsultasikan dengan dokter untuk pemeriksaan lebih lanjut.")
    else:
        st.success(f"✅ Hasil menunjukkan **RISIKO RENDAH** terkena penyakit jantung.\n\nProbabilitas: {prob:.2%}")
        st.write("Tetap jaga pola hidup sehat dan rutin berolahraga.")

# === CATATAN ===
st.markdown("---")
st.caption("Model: RandomForestClassifier (GridSearchCV, ROC-AUC = 0.9376)")
