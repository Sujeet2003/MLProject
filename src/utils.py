from pathlib import Path
import os, sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging


def save_object(file_path: Path, object):
    try:
        path_dir = os.path.dirname(file_path)

        os.makedirs(path_dir, exist_ok=True)
        with open(file_path, "wb") as file:
            dill.dump(object, file)
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path:Path):
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            logging.info(f"Performing for model: {model}")
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            training_model_score = r2_score(y_train, y_train_pred)
            testing_model_score = r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = testing_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)