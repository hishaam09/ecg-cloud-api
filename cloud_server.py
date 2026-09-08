from fastapi import FastAPI
import numpy as np
import tensorflow as tf

app = FastAPI()

model = tf.keras.models.load_model("ecg_cloud_model.h5")

classes = [
    "Normal ECG",
    "Atrial Fibrillation",
    "Arrhythmia"
]

@app.post("/predict")
def predict(data: dict):

    ecg = np.array(data["ecg_signal"])

    if len(ecg) < 200:
        ecg = np.pad(ecg,(0,200-len(ecg)))

    ecg = ecg[:200]

    ecg = ecg.reshape(1,200,1)

    prediction = model.predict(ecg)

    result = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return {
        "diagnosis": result,
        "confidence": confidence
    }
