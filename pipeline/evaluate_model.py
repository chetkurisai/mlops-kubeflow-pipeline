import numpy as np
from tensorflow import keras

def evaluate_model(model_path: str, test_data: str, test_labels: str) -> float:
    """Evaluate the model and return the mean absolute error"""
    model = keras.models.load_model(model_path)
    X_test = np.load(test_data)
    y_test = np.load(test_labels)
    
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    return mae
