import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
from sklearn.metrics import classification_report

# Load dataset
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Convert labels to binary
y_train = (y_train > 0).astype(int)
y_test = (y_test > 0).astype(int)

# reshape for CNN
X_train = X_train.reshape(X_train.shape[0],200,1)
X_test = X_test.reshape(X_test.shape[0],200,1)

# build lightweight model
model = Sequential()

model.add(Conv1D(16,5,activation='relu',input_shape=(200,1)))
model.add(MaxPooling1D(2))

model.add(Conv1D(32,5,activation='relu'))
model.add(MaxPooling1D(2))

model.add(Flatten())

model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test,y_test)
)

# evaluate
pred = model.predict(X_test)
pred_classes = (pred > 0.5).astype(int)

print(classification_report(y_test,pred_classes))

# save model
model.save("ecg_edge_model.keras")

print("Edge model saved")
