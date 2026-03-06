from fastapi import FastAPI
import numpy as np
import tflite_runtime.interpreter as tflite

app = FastAPI()

interpreter = tflite.Interpreter(model_path="ecg_cloud_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.post("/predict")
def predict(data: dict):

    ecg = np.array(data["ecg_signal"])

    if len(ecg) < 200:
        ecg = np.pad(ecg,(0,200-len(ecg)))

    ecg = ecg[:200]

    ecg = ecg.reshape(1,200,1).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], ecg)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])

    result = "Normal ECG" if prediction[0][0] < 0.5 else "Abnormal ECG"

    return {
        "diagnosis": result,
        "confidence": float(prediction[0][0])
    }
