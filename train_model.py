import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

# Load dataset
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Reshape for CNN input
X_train = X_train.reshape(X_train.shape[0], 200, 1)
X_test = X_test.reshape(X_test.shape[0], 200, 1)

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

# Build model
model = Sequential()

model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(200,1)))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(64))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax'))

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Predictions
predictions = model.predict(X_test)
pred_classes = predictions.argmax(axis=1)
true_classes = y_test.argmax(axis=1)

print(classification_report(true_classes, pred_classes))

# Save trained model
model.save("ecg_cloud_model.h5")

print("Model saved successfully.")
