import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import os

st.set_page_config(page_title="Student Stress Prediction", layout="centered")

st.title("Student Stress Prediction System")
st.write("Isi data berikut untuk memprediksi tingkat stres kamu.")

# LOAD DATASET + TRAIN MODEL 

@st.cache_resource
def train_model():
    df = pd.read_csv("Student Stress Factors (2).csv")

    # rename kolom panjang menjadi singkat
    df = df.rename(columns={
        "Kindly Rate your Sleep Quality 😴": "sleep_quality",
        "How many times a week do you suffer headaches 🤕?": "headache_freq",
        "How would you rate you academic performance 👩‍🎓?": "academic_perf",
        "how would you rate your study load?": "study_load",
        "How many times a week you practice extracurricular activities 🎾?": "extracurricular_freq",
        "How would you rate your stress levels?": "stress_level"
    })

    df = df.dropna().reset_index(drop=True)

    feature_cols = ["sleep_quality", "headache_freq", "academic_perf",
                    "study_load", "extracurricular_freq"]

    X = df[feature_cols]
    y = df["stress_level"].astype(int)

    # scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # evaluasi (print ke terminal / log)
    print("Accuracy:", accuracy_score(y_test, knn.predict(X_test)))
    print(classification_report(y_test, knn.predict(X_test)))

    return knn, scaler, feature_cols


# MODEL DILATIH SEKALI SAJA (CACHE)
model, scaler, feature_cols = train_model()

# FORM INPUT USER
st.header("Masukkan Data Kamu")

sleep_quality = st.number_input("Kualitas tidur (1–5)", 1, 5, 3)
headache_freq = st.number_input("Frekuensi sakit kepala per minggu (0–7)", 0, 30, 1)
academic_perf = st.number_input("Kinerja akademik (1–5)", 1, 5, 3)
study_load = st.number_input("Beban studi (1–5)", 1, 5, 3)
extracurricular_freq = st.number_input("Kegiatan ekstrakurikuler per minggu (0–7)", 0, 30, 1)

# PREDIKSI
if st.button("Prediksi Stress Level"):
    input_df = pd.DataFrame([[
        sleep_quality,
        headache_freq,
        academic_perf,
        study_load,
        extracurricular_freq
    ]], columns=feature_cols)

    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]

    level_desc = {
        1: "Sangat rendah",
        2: "Rendah",
        3: "Menengah",
        4: "Tinggi",
        5: "Sangat tinggi"
    }

    st.markdown("### 🔥 Hasil Prediksi")
    st.markdown(f"**{int(pred)} / 5**")
    st.markdown(f"**Tingkat stres kamu: {level_desc.get(int(pred),'Tidak diketahui')}**")

    st.write("---")
    st.info("Model dan scaler dilatih otomatis dari dataset saat aplikasi dijalankan.")

