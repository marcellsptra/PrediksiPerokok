import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Prediksi & Pelatihan Model Perokok", layout="centered")

# --- Fungsi Pelatihan Model ---
def train_and_save_model():
    """
    Fungsi untuk memuat data dari smoking.csv, melatih model,
    dan menyimpannya sebagai file .pkl.
    """
    # Langkah 1: Memuat dan Membersihkan Data
    st.write("Memuat data dari `smoking.csv`...")
    try:
        df = pd.read_csv("smoking.csv", sep=';') # Pengguna mengunggah file dengan pemisah ;
        df.columns = [col.replace(' ', '_').lower() for col in df.columns]
        df['gender'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
        df.drop("id", axis=1, inplace=True, errors='ignore') # Hapus kolom ID jika ada
    except FileNotFoundError:
        st.error("File `smoking.csv` tidak ditemukan. Pastikan file tersebut ada di folder yang sama.")
        return
    except Exception as e:
        st.error(f"Gagal memuat atau memproses CSV: {e}")
        return

    st.write("Data berhasil dimuat. Melakukan prapemrosesan...")

    # Langkah 2: Mendefinisikan Fitur (X) dan Target (y)
    X = df.drop('smoking', axis=1)
    y = df['smoking']
    expected_columns = X.columns.tolist()

    # Langkah 3: Membagi Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.write("Data dibagi menjadi data latih (80%) dan data uji (20%).")

    # Langkah 4: Melatih Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    st.write("Scaler (StandardScaler) berhasil dilatih.")

    # Langkah 5: Melatih Model Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    st.write("Model (RandomForestClassifier) berhasil dilatih.")

    # Langkah 6: Evaluasi Model
    y_pred = rf_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    st.metric(label="Akurasi Model pada Data Uji", value=f"{accuracy:.2%}")

    # Langkah 7: Menyimpan Model, Scaler, dan Kolom
    joblib.dump(rf_model, "random_forest_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(expected_columns, "expected_columns.pkl")

    st.success("🎉 Model, Scaler, dan daftar kolom berhasil dilatih dan disimpan sebagai file `.pkl`!")
    st.info("Aplikasi sekarang siap untuk melakukan prediksi dengan model yang baru.")


# --- Fungsi untuk Memuat Sumber Daya Prediksi ---
@st.cache_resource
def load_prediction_resources():
    """
    Memuat model, scaler, dan kolom yang sudah ada untuk prediksi.
    """
    if not all(os.path.exists(f) for f in ["random_forest_model.pkl", "scaler.pkl", "expected_columns.pkl"]):
        return None, None, None
    
    model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("expected_columns.pkl")
    return model, scaler, columns

# --- Antarmuka Aplikasi Streamlit ---
st.title("🚬 Aplikasi Prediksi Status Perokok")
st.write("Aplikasi ini dapat melakukan prediksi status perokok dan juga melatih ulang model dari data mentah.")

# Sidebar untuk Kontrol
st.sidebar.title("Panel Kontrol")
st.sidebar.write("Gunakan panel ini untuk melatih ulang model atau memasukkan data untuk prediksi.")

# Bagian Pelatihan Model
st.sidebar.header("1. Pelatihan Model")
if st.sidebar.button("Latih Ulang Model dari `smoking.csv`"):
    with st.spinner('Sedang melatih model... Proses ini mungkin memakan waktu beberapa saat.'):
        train_and_save_model()
    # Hapus cache agar sumber daya prediksi dimuat ulang
    st.cache_resource.clear()

st.sidebar.divider()

# Bagian Prediksi
st.sidebar.header("2. Prediksi Status Perokok")

# Memuat sumber daya untuk prediksi
rf_model, scaler, expected_columns = load_prediction_resources()

if rf_model is None:
    st.warning("Model belum tersedia. Latih model terlebih dahulu menggunakan tombol di sidebar.")
else:
    st.sidebar.write("Masukkan data pemeriksaan untuk prediksi:")
    
    input_dict = {}
    # Membuat input field secara dinamis berdasarkan kolom yang diharapkan
    for col in expected_columns:
        label = col.replace('_', ' ').title()
        if col == 'gender':
            input_dict[col] = st.sidebar.selectbox(label, ('Male', 'Female'))
        else:
            # Menggunakan nilai rata-rata dari data asli sebagai default untuk pengalaman pengguna yang lebih baik
            input_dict[col] = st.sidebar.number_input(label, value=0.0, format="%.2f")

    # Tombol Prediksi
    if st.sidebar.button("Buat Prediksi"):
        # Konversi input ke DataFrame
        input_dict['gender'] = 1 if input_dict['gender'] == 'Male' else 0
        input_df = pd.DataFrame([input_dict], columns=expected_columns)

        # Scaling input
        scaled_input = scaler.transform(input_df)

        # Prediksi
        prediction = rf_model.predict(scaled_input)
        prediction_proba = rf_model.predict_proba(scaled_input)

        # Tampilkan Hasil
        st.subheader("Hasil Prediksi")
        smoker_status = "Perokok" if prediction[0] == 1 else "Bukan Perokok"
        
        if smoker_status == "Perokok":
            st.error(f"**Status Terprediksi: {smoker_status}**")
        else:
            st.success(f"**Status Terprediksi: {smoker_status}**")

        st.write("Probabilitas:")
        st.write(f"Bukan Perokok: **{prediction_proba[0][0]:.2%}**")
        st.write(f"Perokok: **{prediction_proba[0][1]:.2%}**")
        
        st.subheader("Data Input yang Digunakan:")
        st.write(input_df)
