from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import NamedTuple

def data_preprocessing() -> NamedTuple("Outputs", [
    ("train_data", str), ("test_data", str), ("train_labels", str), ("test_labels", str)
]):
    """Preprocess data and split into train and test sets"""
    data = load_diabetes()
    X = data.data
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    np.save("pipeline/train_data.npy", X_train)
    np.save("pipeline/test_data.npy", X_test)
    np.save("pipeline/train_labels.npy", y_train)
    np.save("pipeline/test_labels.npy", y_test)
    
    return ("pipeline/train_data.npy", "pipeline/test_data.npy", "pipeline/train_labels.npy", "pipeline/test_labels.npy")
