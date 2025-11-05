import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- 1. Konfigurasi Halaman & Memuat Model ---
st.set_page_config(
    page_title="Prediksi Risiko Jantung",
    page_icon="❤️",
    layout="wide"
)

# Fungsi untuk memuat model dari file pickle
@st.cache_resource
def load_model():
    try:
        # Pastikan nama file ini sesuai dengan file model terbaik Anda
        with open('random_forest_model_terbaik.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("File model 'random_forest_model_terbaik.pkl' tidak ditemukan. Pastikan file berada di direktori yang sama.")
        return None

model = load_model()

# --- 2. Judul dan Deskripsi Aplikasi ---
st.title("❤️ Alat Prediksi Risiko Penyakit Jantung")
st.markdown("""
Aplikasi ini membantu memprediksi seberapa besar risiko seseorang menderita penyakit jantung
berdasarkan data klinis Anda. **Hasil ini bukan diagnosis, selalu konsultasikan dengan dokter.**
""")

st.divider()

if model is not None:
    # --- 3. INPUT FORM BERDASARKAN PSIKOLOGI UI/UX ---

    # Menggunakan tab untuk mengelompokkan input (Prinsip Gestalt: Proximity & Similarity)
    # Ini mengurangi beban kognitif (Hick's Law)
    tab1, tab2, tab3 = st.tabs(["Informasi Dasar", "Hasil Tes Darah & ECG", "Gejala & Gaya Hidup"])

    # ----------------------------------------------------
    # TAB 1: Informasi Dasar
    # ----------------------------------------------------
    with tab1:
        st.subheader("Data Demografi & Fisik")
        colA, colB = st.columns(2)
        
        with colA:
            # Menggunakan label bahasa awam
            age = st.slider("Usia Anda (Tahun)", 20, 90, 50, help="Usia Anda saat ini.")
            
            # Radio button dengan label yang sangat jelas
            sex = st.radio("Jenis Kelamin", 
                           options=[1, 0], 
                           format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan",
                           help="Pilih jenis kelamin Anda. Catatan: Model ini dilatih pada representasi gender binary.")
        
        with colB:
            # Mengubah Resting BP s menjadi "Tekanan Darah"
            resting_bp_s = st.number_input("Tekanan Darah Sistolik Saat Istirahat (mmHg)", 90, 200, 120, help="Tekanan darah sistolik (angka atas) saat Anda sedang istirahat.")
            
            # Mengubah Max Heart Rate menjadi "Detak Jantung Maksimal"
            max_heart_rate = st.number_input("Detak Jantung Maksimal yang Pernah Dicapai", 60, 220, 150, help="Detak jantung tertinggi yang pernah tercatat selama tes (misalnya, treadmill).")
            
            # Hitung fitur hasil rekayasa
            max_heart_rate_per_age = max_heart_rate / age


    # ----------------------------------------------------
    # TAB 2: Hasil Tes Darah & ECG
    # ----------------------------------------------------
    with tab2:
        st.subheader("Hasil Pemeriksaan Laboratorium & Jantung")
        colC, colD = st.columns(2)
        
        with colC:
            cholesterol = st.number_input("Tingkat Kolesterol (mg/dL)", 100, 600, 200, help="Nilai kolesterol serum total Anda.")
            
            # Jargon medis diubah ke bahasa awam dengan penjelasan eksplisit
            fasting_blood_sugar = st.radio("Gula Darah Puasa > 120 mg/dL?", 
                                           options=[1, 0], 
                                           format_func=lambda x: "Ya, di atas 120 mg/dL" if x == 1 else "Tidak, di bawah 120 mg/dL",
                                           help="Apakah hasil gula darah puasa Anda melebihi 120 mg/dL? (Indikasi diabetes).")
            
        with colD:
            # Mengubah Resting ECG
            resting_ecg_options = {
                0: "Normal", 
                1: "Abnormalitas Gelombang ST-T", 
                2: "Kemungkinan atau Definitif Ventrikular Hipertrofi"
            }
            resting_ecg_value = st.selectbox("Hasil Elektrokardiogram (ECG) Saat Istirahat", 
                                             options=list(resting_ecg_options.keys()),
                                             format_func=lambda x: resting_ecg_options[x],
                                             help="Hasil tes ECG Anda. (0=Normal)")
            resting_ecg = resting_ecg_value


    # ----------------------------------------------------
    # TAB 3: Gejala & Gaya Hidup
    # ----------------------------------------------------
    with tab3:
        st.subheader("Gejala dan Indikator Stres Jantung")
        colE, colF = st.columns(2)
        
        with colE:
            # Mengubah chest pain type
            chest_pain_options = {
                0: "Asimtomatik (Tidak Ada Nyeri)",
                1: "Angina Tipikal (Nyeri Khas)", 
                2: "Angina Atipikal (Nyeri Tidak Khas)", 
                3: "Nyeri Non-Anginal"
            }
            chest_pain_type_value = st.selectbox("Tipe Nyeri Dada", 
                                                 options=list(chest_pain_options.keys()),
                                                 format_func=lambda x: chest_pain_options[x],
                                                 help="Klasifikasi nyeri dada yang Anda alami.")
            chest_pain_type = chest_pain_type_value
            
            # Mengubah exercise angina
            exercise_angina = st.radio("Apakah Anda mengalami Nyeri Dada Saat Berolahraga?", 
                                       options=[1, 0], 
                                       format_func=lambda x: "Ya" if x == 1 else "Tidak",
                                       help="Indikasi angina yang dipicu oleh aktivitas fisik.")

        with colF:
            # Mengubah Oldpeak
            oldpeak = st.slider("Depresi Segmen ST Relatif Saat Istirahat (Oldpeak)", 0.0, 6.2, 1.0, 0.1, help="Penurunan segmen ST yang diukur pada elektrokardiogram setelah berolahraga.")
            
            # Mengubah ST Slope
            st_slope_options = {
                0: "Upsloping (Naik)", 
                1: "Flat (Datar)", 
                2: "Downsloping (Menurun)"
            }
            # Catatan: Model Anda dilatih dengan nilai 0, 1, 2, 3. Saya membatasi ke 0-2 untuk kesederhanaan.
            st_slope_value = st.selectbox("Kemiringan Segmen ST Saat Latihan", 
                                         options=list(st_slope_options.keys()),
                                         format_func=lambda x: st_slope_options[x],
                                         help="Bentuk segmen ST saat tes stres olahraga.")
            st_slope = st_slope_value


    # --- 4. Tombol dan Logika Prediksi ---
    st.divider()
    
    # Kumpulkan data input dalam urutan yang benar
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
        'max_heart_rate_per_age': max_heart_rate_per_age
    }
    
    # Urutan kolom yang harus SAMA PERSIS dengan X_train
    final_features = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol', 
                      'fasting blood sugar', 'resting ecg', 'max heart rate', 
                      'exercise angina', 'oldpeak', 'ST slope', 'max_heart_rate_per_age']

    features = pd.DataFrame([data_input], columns=final_features)
    
    # Tombol Prediksi (Affordance: Jelas fungsinya)
    if st.button("🔴 Cek Risiko Saya Sekarang", type="primary", use_container_width=True):
        
        # Prediksi
        prediction_proba = model.predict_proba(features)[:, 1]
        prediction = model.predict(features)[0]

        st.subheader("✅ Hasil Analisis Risiko")
        
        risk_percent = prediction_proba[0] * 100
        
        # Feedback Visual Kuat
        if prediction == 1:
            st.warning(f"## ❗ RISIKO TINGGI: {risk_percent:.2f}%")
            st.error("Berdasarkan data Anda, model memprediksi **kemungkinan tinggi** menderita penyakit jantung.")
            st.markdown("---")
            st.markdown("### Tindakan Selanjutnya yang Disarankan:")
            st.markdown("* **Segera** konsultasi dan lakukan pemeriksaan lebih lanjut dengan dokter spesialis jantung.")
            st.markdown("* Pantau dan kontrol faktor risiko seperti tekanan darah dan kolesterol.")
        else:
            st.success(f"## ✅ RISIKO RENDAH: {risk_percent:.2f}%")
            st.info("Model memprediksi **kemungkinan rendah** menderita penyakit jantung.")
            st.markdown("---")
            st.markdown("### Tindakan Selanjutnya yang Disarankan:")
            st.markdown("* Pertahankan gaya hidup sehat, seperti pola makan dan olahraga teratur.")
            st.markdown("* Tetap lakukan pemeriksaan kesehatan rutin (Medical Check-Up).")
            
        st.caption("⚠️ **PENTING:** Aplikasi ini adalah alat bantu prediksi. Selalu ikuti nasihat dan diagnosis dari tenaga medis profesional.")
