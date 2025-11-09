import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Konfigurasi Halaman (Bersih & Elegan) ---
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    layout="centered", # Menggunakan layout centered untuk tampilan yang bersih
    initial_sidebar_state="expanded"
)

# --- Judul dan Deskripsi Aplikasi ---
st.title("💖 Prediksi Risiko Penyakit Jantung (Heart Disease)")
st.markdown("""
Aplikasi ini memprediksi kemungkinan seseorang menderita penyakit jantung
berdasarkan data klinis, menggunakan model Machine Learning **Random Forest Classifier**
yang telah dilatih. **Disclaimer:** Alat ini hanya untuk tujuan informasi dan bukan pengganti saran medis profesional.
""")

st.divider()

# --- Muat Model Terbaik (random_forest_model.pkl) ---
model_filename = 'random_forest_model.pkl'

if os.path.exists(model_filename):
    try:
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)
        st.sidebar.success("✅ Model Prediksi berhasil dimuat.")
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        model = None
else:
    st.error(f"❌ File model '{model_filename}' tidak ditemukan di direktori saat ini. Pastikan Anda sudah menjalankan langkah penyimpanan model.")
    model = None

# --- Fungsi untuk Input Data Pengguna (User-Friendly) ---
def user_input_features():
    st.sidebar.header("📝 Input Data Pasien")

    # Referensi Kolom dari Data Anda:
    # age, sex, chest pain type, resting bp s, cholesterol, fasting blood sugar,
    # resting ecg, max heart rate, exercise angina, oldpeak, ST slope, max_heart_rate_per_age

    # Catatan: Kolom 'max_heart_rate_per_age' akan dihitung, jadi hanya 11 input utama.

    # 1. Age (float/int)
    age = st.sidebar.slider('Usia (Age)', min_value=20, max_value=80, value=50, step=1)

    # 2. Sex (0=Perempuan, 1=Laki-laki)
    sex_map = {"Laki-laki": 1, "Perempuan": 0}
    sex_label = st.sidebar.selectbox('Jenis Kelamin (Sex)', list(sex_map.keys()))
    sex = sex_map[sex_label]

    # 3. Chest Pain Type (1-4)
    cp_map = {
        "1: Angina Khas (Typical Angina)": 1,
        "2: Angina Atipikal (Atypical Angina)": 2,
        "3: Nyeri Non-Angina (Non-Anginal Pain)": 3,
        "4: Asimtomatik (Asymptomatic)": 4
    }
    cp_label = st.sidebar.selectbox('Tipe Nyeri Dada (Chest Pain Type)', list(cp_map.keys()))
    cp = cp_map[cp_label]

    # 4. Resting Blood Pressure (resting bp s)
    rbp = st.sidebar.number_input('Tekanan Darah Istirahat (Resting BP - mm Hg)', min_value=80, max_value=200, value=130, step=1)

    # 5. Cholesterol (cholesterol) - Ingat ada imputasi median (237.0) untuk nilai 0
    chol = st.sidebar.number_input('Kolesterol Serum (mg/dl)', min_value=80, max_value=603, value=237, step=1)

    # 6. Fasting Blood Sugar (fasting blood sugar) (>120 mg/dl is 1, else 0)
    fbs_map = {"Tidak > 120 mg/dl (0)": 0, "Ya, > 120 mg/dl (1)": 1}
    fbs_label = st.sidebar.selectbox('Gula Darah Puasa > 120 mg/dl?', list(fbs_map.keys()))
    fbs = fbs_map[fbs_label]

    # 7. Resting Electrocardiographic Results (resting ecg) (0, 1, 2)
    recg_map = {
        "0: Normal": 0,
        "1: Kelainan Gelombang ST-T Ringan": 1,
        "2: Hipertrofi Ventrikel Kiri": 2
    }
    recg_label = st.sidebar.selectbox('Hasil EKG Istirahat (Resting ECG)', list(recg_map.keys()))
    recg = recg_map[recg_label]

    # 8. Maximum Heart Rate Achieved (max heart rate)
    mhr = st.sidebar.number_input('Detak Jantung Maks. (Max Heart Rate)', min_value=71, max_value=202, value=150, step=1)

    # 9. Exercise Induced Angina (exercise angina) (1=Yes, 0=No)
    eia_map = {"Tidak (0)": 0, "Ya (1)": 1}
    eia_label = st.sidebar.selectbox('Angina Akibat Olahraga (Exercise Angina)', list(eia_map.keys()))
    eia = eia_map[eia_label]

    # 10. Oldpeak (oldpeak)
    op = st.sidebar.number_input('Oldpeak (Depresi ST Relatif terhadap Istirahat)', min_value=-2.6, max_value=6.2, value=0.6, step=0.1, format="%.2f")

    # 11. ST Slope (ST slope) (1, 2, 3)
    st_slope_map = {
        "1: Naik (Upsloping)": 1,
        "2: Datar (Flat)": 2,
        "3: Turun (Downsloping)": 3
    }
    st_slope_label = st.sidebar.selectbox('Slope ST saat Olahraga (ST Slope)', list(st_slope_map.keys()))
    st_slope = st_slope_map[st_slope_label]

    # 12. max_heart_rate_per_age (fitur hasil rekayasa/engineering)
    mhr_per_age = mhr / age

    data = {
        'age': age,
        'sex': sex,
        'chest pain type': cp,
        'resting bp s': rbp,
        'cholesterol': chol,
        'fasting blood sugar': fbs,
        'resting ecg': recg,
        'max heart rate': mhr,
        'exercise angina': eia,
        'oldpeak': op,
        'ST slope': st_slope,
        'max_heart_rate_per_age': mhr_per_age
    }
    features = pd.DataFrame(data, index=[0])

    # Tampilkan input yang telah diproses di main page (elegan)
    st.subheader("Data Pasien yang Diinput:")
    st.dataframe(features.iloc[:, :-1]) # Tampilkan semua kecuali kolom 'max_heart_rate_per_age'

    return features

# --- Jalankan Fungsi Input ---
input_df = user_input_features()

st.divider()

# --- Tampilkan Tombol Prediksi ---
if model is not None:
    if st.button("Tekan untuk Prediksi Risiko"):
        # Lakukan prediksi
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader("Hasil Prediksi")

        # Visualisasi Hasil dengan Metrik (User-Friendly & Elegan)
        risk_score = prediction_proba[0][1] * 100
        no_risk_score = prediction_proba[0][0] * 100

        # Tampilkan Prognosis
        if prediction[0] == 1:
            st.error(f"🚨 Hasil: **Risiko Tinggi Penyakit Jantung**")
            st.balloons()
        else:
            st.success(f"💚 Hasil: **Risiko Rendah Penyakit Jantung**")

        # Tampilkan Probabilitas
        st.metric(label="Probabilitas Penyakit Jantung",
                  value=f"{risk_score:.2f} %",
                  delta=f"{no_risk_score:.2f} % Kemungkinan Tidak Ada Risiko",
                  delta_color="inverse")

        st.markdown(f"""
        * **Probabilitas Risiko (Target=1):** **{risk_score:.2f}%**
        * **Probabilitas Tidak Ada Risiko (Target=0):** **{no_risk_score:.2f}%**
        """)

    st.caption("Akurasi model terbaik (Random Forest) di data test: Recall = 0.89, Precision = 0.89, ROC AUC = 0.93")
# --- Penutup ---
st.divider()
st.markdown("Dibangun dengan ❤️ menggunakan Streamlit.")
