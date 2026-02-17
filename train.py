from sklearn.linear_model import LinearRegression
from utils import generate_dummy_data, print_model_performance


def train_model():
    # Get data
    X_train, X_test, y_train, y_test = generate_dummy_data()

    # Create model
    model = LinearRegression()

    # Train model
    model.fit(X_train, y_train)

    print("Model training completed!")

    # Evaluate model
    print_model_performance(model, X_test, y_test)

    return model


if __name__ == "__main__":
    train_model()