import streamlit as st
import pandas as pd
import joblib

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Prediksi Status Merokok", layout="wide")

# Judul Aplikasi
st.title("Prediksi Status Merokok 🚬")
st.write("Aplikasi ini memprediksi apakah seseorang perokok atau bukan berdasarkan data pemeriksaan kesehatan.")

# Fungsi untuk memuat model
@st.cache_resource
def load_resources():
    try:
        rf_model = joblib.load("random_forest_model.pkl")
        scaler = joblib.load("scaler.pkl")
        expected_columns = joblib.load("expected_columns.pkl")
        return rf_model, scaler, expected_columns
    except FileNotFoundError as e:
        st.error(f"Error: File model tidak ditemukan. Pastikan file .pkl ada di direktori yang sama. Detail: {e}")
        st.stop()

# Memuat model, scaler, dan kolom
rf_model, scaler, expected_columns = load_resources()


# Sidebar untuk input pengguna
st.sidebar.header("Masukkan Data Pemeriksaan:")

def user_input_features():
    input_dict = {}
    # Loop melalui setiap kolom yang diharapkan untuk membuat input field
    for col in expected_columns:
        label = col.replace('_', ' ').replace('(cm)', ' (cm)').replace('(kg)', ' (kg)').title()
        if col == 'gender':
            # Input khusus untuk gender
            input_dict[col] = st.sidebar.selectbox(label, ('Male', 'Female'))
        elif col in ['hearing(left)', 'hearing(right)', 'oral', 'dental_caries', 'tartar']:
             # Input untuk fitur kategorikal lainnya
            input_dict[col] = st.sidebar.selectbox(label, (1, 0), format_func=lambda x: 'Ya' if x == 1 else 'Tidak')
        else:
            # Input numerik untuk fitur lainnya
            input_dict[col] = st.sidebar.number_input(label, value=0.0, format="%.2f")

    # Konversi input gender ke format numerik (Male=1, Female=0)
    input_dict['gender'] = 1 if input_dict['gender'] == 'Male' else 0
    
    # Buat DataFrame dari input
    data = pd.DataFrame(input_dict, index=[0])
    return data

input_df = user_input_features()

# Tampilkan data input di halaman utama
st.subheader("Data Input Anda:")
st.write(input_df)

# Tombol untuk melakukan prediksi
if st.sidebar.button("Prediksi"):
    try:
        # Pastikan urutan kolom sesuai dengan yang diharapkan oleh model
        input_df = input_df[expected_columns]

        # Lakukan scaling pada data input
        scaled_input = scaler.transform(input_df)

        # Lakukan prediksi
        prediction = rf_model.predict(scaled_input)
        prediction_proba = rf_model.predict_proba(scaled_input)

        # Tampilkan hasil prediksi
        st.subheader("Hasil Prediksi:")
        smoker_status = "Perokok" if prediction[0] == 1 else "Bukan Perokok"
        
        if smoker_status == "Perokok":
            st.warning(f"**Status: {smoker_status}**")
        else:
            st.success(f"**Status: {smoker_status}**")

        # Tampilkan probabilitas
        st.write("Probabilitas:")
        st.write(f"Bukan Perokok: **{prediction_proba[0][0]:.2%}**")
        st.write(f"Perokok: **{prediction_proba[0][1]:.2%}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
