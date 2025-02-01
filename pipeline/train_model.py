import numpy as np
import tensorflow as tf
from tensorflow import keras

def train_model(train_data: str, train_labels: str) -> str:
    """Train a simple neural network model"""
    X_train = np.load(train_data)
    y_train = np.load(train_labels)
    
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)
    model.save("pipeline/model.h5")
    return "pipeline/model.h5"
