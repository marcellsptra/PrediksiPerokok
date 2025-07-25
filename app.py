
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Prediksi Status Merokok")

# Load models
dt_model = joblib.load("decision_tree_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("expected_columns.pkl")

st.write("Masukkan data input:")
input_dict = {col: st.number_input(col, value=0.0) for col in expected_columns}
input_df = pd.DataFrame([input_dict])

if st.button("Prediksi"):
    scaled_input = scaler.transform(input_df)
    pred = rf_model.predict(scaled_input)[0]
    st.success(f"Prediksi Status Merokok: {'Ya' if pred == 1 else 'Tidak'}")
