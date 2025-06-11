import mlflow
import os
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))


def train_model(X_train, y_train):
    """
    Train a Random Forest Regressor model on the training data.
    Returns:
    model: The trained Random Forest Regressor model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test data.
    Parameters:
    model: The trained Random Forest Regressor model.
    X_test (pd.DataFrame): The test features.
    y_test (pd.Series): The target variable for testing.

    Returns:
    float: The RMSE of the model on the test data.
    """
    predictions = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)
    return rmse


def save_model(model, filename):
    """
    Save the trained model to a file.
    Parameters:
    model: The trained Random Forest Regressor model.
    filename (str): The filename to save the model to.
    """
    import pickle

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)


def track_experiment(X_train, y_train):

    mlflow.set_experiment("sales-predictor-experiment")
    """
    Track the experiment using MLflow.
    """
    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_samples", len(X_train))
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        rmse = evaluate_model(model, X_train, y_train)
        mlflow.log_metric("rmse", rmse)
        return model


