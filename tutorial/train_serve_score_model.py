import os
import sys
import warnings
import logging

import numpy as np
import pandas as pd
from urllib.parse import urlparse

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    
    return rmse, mae, r2

def make_model_input(data):
    # Split the data into training and test sets
    train, test = train_test_split(data)
    
    train_x = train.drop(['quality'], axis=1)
    test_x = test.drop(['quality'], axis=1)
    train_y = train[['quality']]
    test_y = test[['quality']]

    return train_x, test_x, train_y, test_y
    
def fit_and_predict_model(train_set, test_set):
    train_x, train_y = train_set
    test_x, test_y = test_set
    
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=2)
    lr.fit(train_x, train_y)
    
    prediction = lr.predict(test_x)

    rmse, mae, r2 = eval_metrics(test_y, prediction)
    
    print(f"Elasticnet model (alpha={alpha}, l1_ratio={l1_ratio}):")
    print(f"RMSE: {rmse}" % rmse)
    print(f"MAE: {mae}" % mae)
    print(f"R2: {r2}" % r2)
    
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('r2', r2)
    mlflow.log_metric('mae', mae)
    
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    
    # Model registry does not work with file store
    if tracking_url_type_store != "file":
        # Register the model
        mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetModel")
    else:
        mlflow.sklearn.log_model(lr, "model")
    
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    np.random.seed(2)
    
    csv_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try: 
        data = pd.read_csv(csv_url, sep=';')
    except Exception as e:
        logger.exception(f"Unable to download training & test CSV, check your internet connection. Error: {e}")
    
    train_x, test_x, train_y, test_y = make_model_input(data)
    
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    with mlflow.start_run():
        fit_and_predict_model(
            train_set=(train_x, train_y),
            test_set=(test_x, test_y)
        )