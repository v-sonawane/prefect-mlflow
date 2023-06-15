from prefect import task, flow

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

from datetime import timedelta

import numpy as np
import pandas as pd

import mlflow
import requests

@task
def fetch_data():
    csv_url = "https://raw.githubusercontent.com/aniruddhachoudhury/Red-Wine-Quality/master/winequality-red.csv"
    data = pd.read_csv(csv_url, sep=",")
    return data


@task
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    return rmse, mae

@flow
def train_model(mlflow_experiment_id, alpha=0.5, l1_ratio=0.5):
    
    mlflow.set_tracking_uri("http://localhost:5000")
    data=fetch_data()
    target=data["quality"]
    data=data.drop(["quality"],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    
    with mlflow.start_run(experiment_id=mlflow_experiment_id):
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(X_train, y_train)
        predicted_qualities = lr.predict(X_test)
        (rmse, mae) = eval_metrics(y_test, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "model")


if __name__ == "__main__":
    experiment_name = "wine-quality-test"
    mlflow_experiment_id = mlflow.create_experiment(experiment_name)
    mlflow_experiment_id = experiment.experiment_id
    
    train_model(mlflow_experiment_id,alpha=0.3, l1_ratio=0.3)
    train_model(mlflow_experiment_id, alpha=0.5, l1_ratio=0.5)
