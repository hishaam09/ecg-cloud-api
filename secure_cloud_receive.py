from Crypto.Cipher import AES
import numpy as np

# Same key used during encryption
key = b'0123456789abcdef'

# Load encrypted ECG
with open("encrypted_ecg.bin","rb") as f:
    encrypted_data = f.read()

cipher = AES.new(key, AES.MODE_EAX, nonce=b'1234567890123456')

# Decrypt
decrypted_data = cipher.decrypt(encrypted_data)

# Convert back to ECG signal
ecg_signal = np.frombuffer(decrypted_data)

print("ECG decrypted successfully.")
print("Recovered signal length:", len(ecg_signal))
