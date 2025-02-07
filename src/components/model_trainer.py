import os, sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_training_model(self, train, test):
        try:
            logging.info("Spliting training and test input data...")
            X_train, y_train, X_test, y_test = (
                train[:, :-1],
                train[:, -1],
                test[:, :-1],
                test[:, -1]    
            )
            models = {
                'Linear Regressor': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(),    
                'Random Forest': RandomForestRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
            }

            params = {
                'Linear Regressor': {

                },
                'Decision Tree': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                'Random Forest': {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                'Gradient Boosting': {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.8, .85, .9],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
            }

            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No Any Best Model Found!!!")
            logging.info("Best model found for the given train and test datasets.")
            save_object(file_path=self.model_trainer_config.trained_model_file_path, object=best_model)
            logging.info("Best model got saved into the directory.")

            predicted = best_model.predict(X_test)
            r2_score_val = r2_score(y_test, predicted)
            return r2_score_val
        
        except Exception as e:
            raise Exception(e, sys)