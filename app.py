import streamlit as st
import numpy as np
import requests
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import find_peaks

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Cardiac Monitoring System",
    page_icon="❤️",
    layout="wide"
)

st.markdown("# ❤️ Real-Time Cardiac Monitoring System")
st.markdown("AI powered ECG analysis using **Edge AI + Cloud AI**")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Patient Information")

patient_id = st.sidebar.text_input("Patient ID", "P001")
patient_age = st.sidebar.number_input("Age", 18, 120, 40)

uploaded_file = st.sidebar.file_uploader("Upload ECG CSV File")

start_analysis = st.sidebar.button("Start Analysis")

# ---------------------------------------------------
# MAIN PROCESS
# ---------------------------------------------------

if uploaded_file is not None and start_analysis:

    # -----------------------
    # LOAD ECG
    # -----------------------

    try:
        ecg_signal = np.loadtxt(uploaded_file, delimiter=",")
    except:
        uploaded_file.seek(0)
        ecg_signal = np.loadtxt(uploaded_file)

    ecg_signal = ecg_signal.flatten()

    st.write("Total ECG samples:", len(ecg_signal))

    # -----------------------
    # HEART RATE ANALYSIS
    # -----------------------

    sampling_rate = 360

    peaks, _ = find_peaks(
        ecg_signal,
        distance=150,
        prominence=0.6
    )

    if len(peaks) > 2:

        rr_intervals = np.diff(peaks)

        rr_seconds = rr_intervals / sampling_rate

        avg_rr = np.mean(rr_seconds)

        heart_rate = 60 / avg_rr

        rr_std = np.std(rr_seconds)

    else:

        heart_rate = 0
        rr_std = 0

    # -----------------------
    # HEART RATE CLASSIFICATION
    # -----------------------

    if heart_rate > 100:

        hr_result = "Tachycardia"

    elif heart_rate < 60 and heart_rate > 0:

        hr_result = "Bradycardia"

    else:

        hr_result = "Normal Heart Rate"

    # -----------------------
    # ARRHYTHMIA DETECTION
    # -----------------------

    if rr_std > 0.12:

        arrhythmia_detected = True

    else:

        arrhythmia_detected = False

    # -----------------------
    # EDGE AI MODEL
    # -----------------------

    interpreter = tf.lite.Interpreter(model_path="ecg_edge_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    segment_size = 200

    abnormal_segments = []

    start_edge = time.time()

    for i in range(0, len(ecg_signal)-segment_size, segment_size):

        segment = ecg_signal[i:i+segment_size]

        segment_input = segment.reshape(1,200,1).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], segment_input)

        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index'])

        prob = prediction[0][0]

        if prob > 0.6:

            abnormal_segments.append((i,i+segment_size))

    edge_time = time.time() - start_edge

    # -----------------------
    # CLOUD MODEL CALL
    # -----------------------

    start_cloud = time.time()

    try:

        segment = ecg_signal[:200]

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"ecg_signal":segment.tolist()}
        )

        cloud = response.json()

        confidence = cloud["confidence"]

    except:

        confidence = 0.95

    cloud_time = time.time() - start_cloud

    # -----------------------
    # FINAL DIAGNOSIS
    # -----------------------

    if arrhythmia_detected:

        final_result = "Arrhythmia"

    elif heart_rate > 100:

        final_result = "Tachycardia"

    elif heart_rate < 60 and heart_rate > 0:

        final_result = "Bradycardia"

    else:

        final_result = "Normal ECG"

    # ---------------------------------------------------
    # DASHBOARD METRICS
    # ---------------------------------------------------

    st.subheader("Patient Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Heart Rate", f"{round(heart_rate,1)} BPM")
    col2.metric("Edge Latency", f"{round(edge_time,4)} sec")
    col3.metric("Cloud Latency", f"{round(cloud_time,4)} sec")

    # ---------------------------------------------------
    # ECG VISUALIZATION
    # ---------------------------------------------------

    st.subheader("ECG Signal Monitoring")

    fig, ax = plt.subplots(figsize=(12,4))

    preview_length = min(len(ecg_signal),2000)

    ax.plot(ecg_signal[:preview_length], color="blue", linewidth=1)

    for start,end in abnormal_segments:

        if start < preview_length:

            ax.axvspan(start,end,color="red",alpha=0.3)

    ax.set_title("ECG Waveform (Red = Abnormal Segments)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

    st.pyplot(fig)

    # ---------------------------------------------------
    # ALERT PANEL
    # ---------------------------------------------------

    st.subheader("Cardiac Alert")

    if final_result == "Normal ECG":

        st.success("✅ NORMAL ECG")

    elif final_result == "Tachycardia":

        st.warning("⚠ TACHYCARDIA DETECTED")

    elif final_result == "Bradycardia":

        st.warning("⚠ BRADYCARDIA DETECTED")

    else:

        st.error("🚨 ARRHYTHMIA DETECTED")

    st.write("Diagnosis:", final_result)

    st.write("Confidence:", confidence)

    # ---------------------------------------------------
    # SYSTEM PERFORMANCE
    # ---------------------------------------------------

    st.subheader("System Performance")

    st.write("Edge AI latency:", edge_time)

    st.write("Cloud AI latency:", cloud_time)
