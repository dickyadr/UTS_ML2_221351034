import streamlit as st
import tensorflow  as tf
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="iris.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Iris Species")
st.write("Masukkan informasi Bunga Iris.")

# Form input pengguna
A = st.number_input("SepalLengthCm (A)", min_value=0.0, max_value=200.0, value=50.0)
B = st.number_input("SepalWidthCm (B)", min_value=0.0, max_value=200.0, value=50.0)
C = st.number_input("PetalLengthCm (C)", min_value=0.0, max_value=200.0, value=50.0)
D = st.number_input("PetalWidthCm (D)", min_value=0.0, max_value=200.0, value=50.0)


if st.button("Rekomendasikan Iris"):
    # Preprocessing input
    input_data = np.array([[A,B,C,D]])
    input_scaled = scaler.transform(input_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_label = np.argmax(prediction)
    crop_name = label_encoder.inverse_transform([predicted_label])[0]

    st.success(f"Rekomendasi Iris: **{crop_name.upper()}**")
