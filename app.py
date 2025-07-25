import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os

# --- Konfigurasi Halaman & Path ---
st.set_page_config(page_title="Prediksi & Pelatihan Model", layout="centered")

# Mendefinisikan path agar lebih rapi dan mudah diubah
MODEL_DIR = 'model'
DATA_PATH = os.path.join(MODEL_DIR, 'dataset.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')

# --- Fungsi Pelatihan Model ---
def train_and_save_model():
    """
    Fungsi untuk memuat data dari model/dataset.csv, melatih model,
    dan menyimpannya sebagai satu file model/model.pkl.
    """
    st.write(f"Memuat data dari `{DATA_PATH}`...")
    try:
        df = pd.read_csv(DATA_PATH, sep=';')
        df.columns = [col.replace(' ', '_').lower() for col in df.columns]
        df['gender'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
        df.drop("id", axis=1, inplace=True, errors='ignore')
    except FileNotFoundError:
        st.error(f"File `{DATA_PATH}` tidak ditemukan. Pastikan file ada di dalam folder 'model'.")
        return
    except Exception as e:
        st.error(f"Gagal memuat atau memproses CSV: {e}")
        return

    st.write("Data berhasil dimuat. Melakukan prapemrosesan...")

    X = df.drop('smoking', axis=1)
    y = df['smoking']
    expected_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.write("Data dibagi menjadi data latih (80%) dan data uji (20%).")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    st.write("Scaler (StandardScaler) berhasil dilatih.")

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    st.write("Model (RandomForestClassifier) berhasil dilatih.")

    y_pred = rf_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    st.metric(label="Akurasi Model pada Data Uji", value=f"{accuracy:.2%}")

    # Menggabungkan semua komponen ke dalam satu dictionary
    model_artifacts = {
        'model': rf_model,
        'scaler': scaler,
        'columns': expected_columns
    }

    # Membuat folder 'model' jika belum ada
    os.makedirs(MODEL_DIR, exist_ok=True)
    # Menyimpan dictionary sebagai satu file .pkl
    joblib.dump(model_artifacts, MODEL_PATH)

    st.success(f"🎉 Model berhasil dilatih dan disimpan di `{MODEL_PATH}`!")
    st.info("Aplikasi sekarang siap untuk melakukan prediksi dengan model yang baru.")


# --- Fungsi untuk Memuat Sumber Daya Prediksi ---
@st.cache_resource
def load_prediction_resources():
    """
    Memuat model, scaler, dan kolom dari satu file model.pkl.
    """
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    
    try:
        model_artifacts = joblib.load(MODEL_PATH)
        model = model_artifacts['model']
        scaler = model_artifacts['scaler']
        columns = model_artifacts['columns']
        return model, scaler, columns
    except Exception as e:
        st.error(f"Gagal memuat file model: {e}")
        return None, None, None

# --- Antarmuka Aplikasi Streamlit ---
st.title("🚬 Aplikasi Prediksi Status Perokok")
st.write("Struktur proyek baru dengan folder `model`.")

st.sidebar.title("Panel Kontrol")

st.sidebar.header("1. Pelatihan Model")
if st.sidebar.button("Latih Ulang Model"):
    with st.spinner('Sedang melatih model...'):
        train_and_save_model()
    st.cache_resource.clear()

st.sidebar.divider()

st.sidebar.header("2. Prediksi Status Perokok")

rf_model, scaler, expected_columns = load_prediction_resources()

if rf_model is None:
    st.warning(f"File `{MODEL_PATH}` tidak ditemukan. Latih model terlebih dahulu.")
else:
    st.sidebar.write("Masukkan data untuk prediksi:")
    
    input_dict = {}
    for col in expected_columns:
        label = col.replace('_', ' ').title()
        if col == 'gender':
            input_dict[col] = st.sidebar.selectbox(label, ('Male', 'Female'))
        else:
            input_dict[col] = st.sidebar.number_input(label, value=0.0, format="%.2f")

    if st.sidebar.button("Buat Prediksi"):
        input_df = pd.DataFrame([input_dict])
        input_df['gender'] = 1 if input_df['gender'][0] == 'Male' else 0
        input_df = input_df[expected_columns]

        scaled_input = scaler.transform(input_df)
        prediction = rf_model.predict(scaled_input)
        prediction_proba = rf_model.predict_proba(scaled_input)

        st.subheader("Hasil Prediksi")
        smoker_status = "Perokok" if prediction[0] == 1 else "Bukan Perokok"
        
        if smoker_status == "Perokok":
            st.error(f"**Status Terprediksi: {smoker_status}**")
        else:
            st.success(f"**Status Terprediksi: {smoker_status}**")

        st.write("Probabilitas:")
        st.write(f"Bukan Perokok: **{prediction_proba[0][0]:.2%}**")
        st.write(f"Perokok: **{prediction_proba[0][1]:.2%}**")
