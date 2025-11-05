import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- 1. Konfigurasi Halaman & Memuat Model ---
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="❤️",
    layout="wide"
)

# Fungsi untuk memuat model dari file pickle
@st.cache_resource # Cache resource agar model hanya dimuat sekali
def load_model():
    try:
        # Nama file harus sesuai dengan file pickle yang sudah Anda simpan
        with open('random_forest_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("File model 'random_forest_model.pkl' tidak ditemukan. Pastikan file berada di direktori yang sama.")
        return None

model = load_model()

# --- 2. Judul dan Deskripsi Aplikasi ---
st.title("❤️ Aplikasi Prediksi Risiko Penyakit Jantung")
st.markdown("""
Aplikasi ini memprediksi kemungkinan seseorang menderita penyakit jantung 
berdasarkan data klinis menggunakan model Machine Learning.
""")

st.divider()

if model is not None:
    # --- 3. Form Input Data Pengguna (Beginner Friendly) ---
    st.header("Masukkan Data Pasien:")

    # Struktur kolom untuk tampilan yang lebih rapi
    col1, col2, col3 = st.columns(3)

    # Input Kolom 1
    with col1:
        age = st.slider("1. Usia (Age)", 20, 90, 50)
        sex = st.radio("2. Jenis Kelamin (Sex)", 
                       options=[1, 0], 
                       format_func=lambda x: "Laki-laki (1)" if x == 1 else "Perempuan (0)")
        
        # Penjelasan Chest Pain Type
        st.info("Tipe Nyeri Dada (Chest Pain Type): 0=Asimtomatik, 1=Angina Tipikal, 2=Angina Atipikal, 3=Non-Anginal")
        chest_pain_type = st.selectbox("3. Tipe Nyeri Dada", options=[0, 1, 2, 3])

    # Input Kolom 2
    with col2:
        resting_bp_s = st.number_input("4. Tekanan Darah Istirahat (Resting BP s)", 90, 200, 120)
        cholesterol = st.number_input("5. Kolesterol Serum (mg/dl)", 100, 600, 200)
        
        # Fasting Blood Sugar (Fasting Blood Sugar > 120 mg/dL = 1, else = 0)
        fasting_blood_sugar = st.radio("6. Gula Darah Puasa > 120 mg/dL (Fasting Blood Sugar)", 
                                       options=[1, 0], 
                                       format_func=lambda x: "Ya (1)" if x == 1 else "Tidak (0)")
        
    # Input Kolom 3
    with col3:
        max_heart_rate = st.number_input("7. Detak Jantung Maksimal (Max Heart Rate)", 60, 220, 150)
        
        # Resting ECG
        st.info("Hasil ECG Istirahat (Resting ECG): 0=Normal, 1=ST-T Wave Abnormality, 2=Hypertrophy")
        resting_ecg = st.selectbox("8. Hasil ECG Istirahat", options=[0, 1, 2])
        
        # Exercise Angina
        exercise_angina = st.radio("9. Angina Akibat Olahraga (Exercise Angina)", 
                                   options=[1, 0], 
                                   format_func=lambda x: "Ya (1)" if x == 1 else "Tidak (0)")

    # Input Oldpeak dan ST Slope diletakkan di bawah atau di kolom lain jika ada 12 fitur
    col4, col5, col6 = st.columns(3)
    with col4:
        oldpeak = st.slider("10. Depresi ST (Oldpeak)", 0.0, 6.2, 1.0, 0.1)
    
    with col5:
        # ST Slope
        st.info("Kemiringan Segmen ST (ST Slope): 0=Upsloping, 1=Flat, 2=Downsloping")
        st_slope = st.selectbox("11. Kemiringan Segmen ST", options=[0, 1, 2, 3])
    
    # Hitung fitur hasil rekayasa (Feature Engineering)
    # Gunakan fitur yang Anda buat: max_heart_rate / age
    max_heart_rate_per_age = max_heart_rate / age
    
    # --- 4. Tombol Prediksi ---
    st.divider()
    
    # Mengumpulkan semua data input ke dalam satu DataFrame untuk prediksi
    data_input = {
        'age': age,
        'sex': sex,
        'chest pain type': chest_pain_type,
        'resting bp s': resting_bp_s,
        'cholesterol': cholesterol,
        'fasting blood sugar': fasting_blood_sugar,
        'resting ecg': resting_ecg,
        'max heart rate': max_heart_rate,
        'exercise angina': exercise_angina,
        'oldpeak': oldpeak,
        'ST slope': st_slope,
        'target': np.nan, # 'target' tidak digunakan untuk prediksi, tapi untuk konsistensi jumlah kolom (jika df.drop dihilangkan dari preprocessing)
        'max_heart_rate_per_age': max_heart_rate_per_age
    }
    
    # Hapus kolom 'target' dari data input sebelum membuat DataFrame jika model Anda dilatih tanpa 'target'
    del data_input['target']
    
    # Membuat DataFrame dengan urutan kolom yang benar (Sangat Penting!)
    # Anda perlu menyesuaikan urutan kolom ini agar SAMA PERSIS dengan X_train saat model dilatih
    # Berdasarkan PDF, urutan kolom Anda setelah FE dan drop 'target' adalah:
    # ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol', 'fasting blood sugar', 'resting ecg', 'max heart rate', 'exercise angina', 'oldpeak', 'ST slope', 'max_heart_rate_per_age']
    
    # Catatan: Karena Anda tidak memberikan urutan akhir kolom X_train, saya menggunakan urutan logis
    # Jika model Anda dilatih dengan 12 fitur (termasuk 'max_heart_rate_per_age'), ini harusnya 12 fitur:
    final_features = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol', 
                      'fasting blood sugar', 'resting ecg', 'max heart rate', 
                      'exercise angina', 'oldpeak', 'ST slope', 'max_heart_rate_per_age']

    # Buat DataFrame dari input
    features = pd.DataFrame([data_input], columns=final_features)
    
    # Logika untuk menampilkan hasil
    if st.button("🔎 Prediksi Risiko"):
        # Prediksi Probabilitas
        prediction_proba = model.predict_proba(features)[:, 1]
        
        # Prediksi Kelas (0 atau 1)
        prediction = model.predict(features)[0]

        st.subheader("📊 Hasil Prediksi")
        
        # Menampilkan Probabilitas
        risk_percent = prediction_proba[0] * 100
        st.metric(label="Tingkat Risiko Penyakit Jantung", 
                  value=f"{risk_percent:.2f}%")

        # Menampilkan Keputusan Akhir
        if prediction == 1:
            st.error("❗ Risiko Tinggi")
            st.markdown("Pasien diprediksi memiliki **risiko tinggi** menderita penyakit jantung. **Sangat dianjurkan** untuk konsultasi lebih lanjut dengan profesional medis.")
        else:
            st.success("✅ Risiko Rendah")
            st.markdown("Pasien diprediksi memiliki **risiko rendah** menderita penyakit jantung.")
            
        st.caption("Disclaimer: Aplikasi ini hanya untuk tujuan demonstrasi dan tidak boleh digunakan sebagai pengganti diagnosis medis profesional.")