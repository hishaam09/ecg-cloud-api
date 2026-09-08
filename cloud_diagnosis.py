import numpy as np
import tensorflow as tf
import time

# Load cloud model
model = tf.keras.models.load_model("ecg_cloud_model.h5")

# Load ECG CSV
ecg_signal = np.loadtxt("ecg_sample.csv")

# Ensure correct length
ecg_signal = ecg_signal[:200]

# reshape for model
ecg_signal = ecg_signal.reshape(1,200,1)

# start timer
start = time.time()

prediction = model.predict(ecg_signal)

end = time.time()

latency = end - start

classes = [
    "Normal ECG",
    "Atrial Fibrillation Detected",
    "Arrhythmia Detected"
]

result = classes[np.argmax(prediction)]

print("Cloud Diagnosis:", result)
print("Confidence:", np.max(prediction))
print("Cloud Inference Time:", latency,"seconds")
