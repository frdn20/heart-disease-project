import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# ========== KONFIGURASI DASAR ==========
st.set_page_config(
    page_title="Heart Disease App",
    page_icon="❤️",
    layout="wide"
)

# ========== SIDEBAR NAVIGASI ==========
st.sidebar.title("🏠 Navigasi")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["📊 Eksplorasi Data (EDA)", "🧠 Prediksi Risiko Penyakit Jantung"]
)

# === LOAD DATA DAN MODEL ===
@st.cache_data
def load_data():
    df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")
    df = df.drop_duplicates()
    df = df[df['resting bp s'] != 0]
    median_chol = df[df['cholesterol'] != 0]['cholesterol'].median()
    df['cholesterol'] = df['cholesterol'].replace(0, median_chol)
    df["max_heart_rate_per_age"] = df["max heart rate"] / df["age"]
    return df

df = load_data()

with open("random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

# ======================================================================
# ==================== 1. HALAMAN EKSPLORASI DATA =====================
# ======================================================================
if menu == "📊 Eksplorasi Data (EDA)":
    st.title("📊 Eksplorasi Data Penyakit Jantung")
    st.write("Gunakan halaman ini untuk melihat karakteristik data dan pola antara fitur dengan risiko penyakit jantung.")

    st.markdown("---")
    st.subheader("Pilih Visualisasi:")
    option = st.selectbox(
        "Pilih jenis analisis yang ingin dilihat:",
        [
            "Distribusi Target",
            "Distribusi Umur per Target",
            "Jenis Kelamin vs Target",
            "Boxplot Fitur Numerik vs Target",
            "Korelasi Heatmap",
            "Outlier Check",
            "Feature Tambahan (max_heart_rate_per_age)"
        ]
    )

    # --- 1. Distribusi Target
    if option == "Distribusi Target":
        st.subheader("Distribusi Target (0 = Tidak Sakit, 1 = Sakit Jantung)")
        fig, ax = plt.subplots()
        sns.countplot(x='target', data=df, ax=ax)
        st.pyplot(fig)

    # --- 2. Distribusi Umur per Target
    elif option == "Distribusi Umur per Target":
        st.subheader("Distribusi Umur berdasarkan Status Penyakit Jantung")
        fig, ax = plt.subplots()
        sns.histplot(data=df, x='age', hue='target', kde=True, ax=ax)
        st.pyplot(fig)

    # --- 3. Jenis Kelamin vs Target
    elif option == "Jenis Kelamin vs Target":
        st.subheader("Distribusi Jenis Kelamin terhadap Penyakit Jantung")
        fig, ax = plt.subplots()
        sns.countplot(x='sex', hue='target', data=df, ax=ax)
        ax.set_xticklabels(['Wanita', 'Pria'])
        st.pyplot(fig)

    # --- 4. Boxplot Numerik
    elif option == "Boxplot Fitur Numerik vs Target":
        st.subheader("Perbandingan Fitur Numerik terhadap Target")
        num_cols = ['age','resting bp s','cholesterol','max heart rate','oldpeak']
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.boxplot(x='target', y=col, data=df, ax=ax)
            st.pyplot(fig)

    # --- 5. Korelasi Heatmap
    elif option == "Korelasi Heatmap":
        st.subheader("Korelasi Antar Fitur terhadap Target")
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(corr, annot=True, cmap='RdYlGn', ax=ax)
        st.pyplot(fig)

    # --- 6. Outlier Check
    elif option == "Outlier Check":
        st.subheader("Pengecekan Outlier pada Fitur Sensitif")
        for col in ['resting bp s', 'cholesterol', 'oldpeak']:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f"Outlier Check: {col}")
            st.pyplot(fig)

    # --- 7. Feature Baru
    elif option == "Feature Tambahan (max_heart_rate_per_age)":
        st.subheader("Rasio Denyut Jantung Maksimum per Usia terhadap Risiko")
        fig, ax = plt.subplots()
        sns.scatterplot(x='max_heart_rate_per_age', y='age', hue='target', data=df, ax=ax)
        st.pyplot(fig)

    st.markdown("---")
    st.info("💡 Gunakan EDA ini untuk memahami bagaimana faktor-faktor seperti umur, tekanan darah, kolesterol, dan aktivitas memengaruhi risiko penyakit jantung.")

# ======================================================================
# ==================== 2. HALAMAN PREDIKSI MODEL =====================
# ======================================================================
elif menu == "🧠 Prediksi Risiko Penyakit Jantung":
    st.title("🧠 Prediksi Risiko Penyakit Jantung")
    st.write("Isi data di bawah untuk mengetahui apakah seseorang berisiko terkena penyakit jantung.")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Umur", 20, 100, 40)
        sex = st.selectbox("Jenis Kelamin", ["Pria", "Wanita"])
        chest_pain = st.selectbox("Tipe Nyeri Dada (1-4)", [1,2,3,4])
        resting_bp = st.number_input("Tekanan Darah Istirahat (mm Hg)", 80, 200, 130)
        cholesterol = st.number_input("Kadar Kolesterol (mg/dl)", 85, 600, 230)

    with col2:
        fasting_blood_sugar = st.selectbox("Gula Darah Puasa > 120 mg/dl", ["Tidak", "Ya"])
        resting_ecg = st.selectbox("Hasil ECG", [0, 1, 2])
        max_heart_rate = st.number_input("Denyut Jantung Maksimum", 60, 220, 150)
        exercise_angina = st.selectbox("Angina Saat Olahraga", ["Tidak", "Ya"])
        oldpeak = st.number_input("Oldpeak (Depresi ST)", -3.0, 6.0, 1.0)
        st_slope = st.selectbox("ST Slope (0-2)", [0,1,2])

    # Buat dataframe input
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

    if st.button("🚀 Prediksi Sekarang"):
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.subheader("📊 Hasil Prediksi")
        if pred == 1:
            st.error(f"⚠️ Hasil menunjukkan **RISIKO TINGGI** terkena penyakit jantung.\n\nProbabilitas: {prob:.2%}")
            st.write("Segera konsultasikan dengan dokter dan jaga pola hidup sehat.")
        else:
            st.success(f"✅ Hasil menunjukkan **RISIKO RENDAH** terkena penyakit jantung.\n\nProbabilitas: {prob:.2%}")
            st.write("Tetap jaga gaya hidup sehat dan rutin berolahraga.")

    st.markdown("---")
    st.caption("Model: RandomForestClassifier (GridSearchCV, ROC-AUC = 0.9376)")
