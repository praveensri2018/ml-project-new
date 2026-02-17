
import numpy as np
from sklearn.model_selection import train_test_split


def generate_dummy_data():
    """
    Generate simple dummy dataset for training
    """
    X = np.array([[1], [2], [3], [4], [5], [6]])
    y = np.array([2, 4, 6, 8, 10, 12])  # y = 2x

    return train_test_split(X, y, test_size=0.2, random_state=42)


def print_model_performance(model, X_test, y_test):
    """
    Print model score
    """
    score = model.score(X_test, y_test)
    print(f"Model Accuracy: {score:.2f}")