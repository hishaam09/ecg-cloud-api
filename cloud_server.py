from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

app = FastAPI()

# Load cloud model
model = tf.keras.models.load_model("ecg_cloud_model.h5")

classes = [
    "Normal ECG",
    "Atrial Fibrillation",
    "Arrhythmia"
]

# Request format
class ECGInput(BaseModel):
    ecg_signal: list


@app.post("/predict")
def predict(data: ECGInput):

    ecg = np.array(data.ecg_signal)

    # pad or trim
    if len(ecg) < 200:
        ecg = np.pad(ecg, (0, 200 - len(ecg)))

    ecg = ecg[:200]

    ecg = ecg.reshape(1, 200, 1)

    prediction = model.predict(ecg)

    result = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return {
        "diagnosis": result,
        "confidence": confidence
    }
