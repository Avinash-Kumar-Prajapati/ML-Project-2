import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    model_path=os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def evaluate_model(self, X_train, Y_train, X_test, Y_test, models, params):
        try:
            result={}
            for modelname, model in models.items():
                param=params[modelname]
                gsc=GridSearchCV(estimator=model, param_grid=param, verbose=3, cv=3)

                gsc.fit(X_train, Y_train)

                model.set_params(**gsc.best_params_)
                model.fit(X_train, Y_train)
                Y_pred=model.predict(X_test)

                r2s=r2_score(Y_test, Y_pred)

                result[modelname]=r2s

            return result

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_model_trainer(self, train_data, test_data):
        try:
            logging.info('Splitting train data and test data')
            X_train, Y_train, X_test, Y_test=(train_data[:,:-1],
                                              train_data[:,-1],
                                              test_data[:,:-1],
                                              test_data[:,-1])
            
            models={
                "LinearRegression":LinearRegression(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "KNN": KNeighborsRegressor(),
                "GradientBoost": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Catboost": CatBoostRegressor(verbose=False),
                "XGBoost": XGBRegressor(),
                "Lasso": Lasso(),
                "Ridge": Ridge()
            }

            params = {
                "LinearRegression": {},

                "DecisionTree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },

                "RandomForest": {
                    'n_estimators': [50, 100, 150, 200, 250, 300],
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },

                "KNN": {
                    'n_neighbors': [3, 5, 7, 9, 11]
                },

                "GradientBoost": {
                    'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.09],
                    'n_estimators': [50, 100, 150, 200, 250],
                    'subsample': [0.5, 0.7, 0.9]
                },

                "AdaBoost": {
                    'learning_rate': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'n_estimators': [50, 100, 150, 200, 250],
                    'loss': ['linear', 'square', 'exponential']
                },

                "Catboost": {
                    'depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.03, 0.05],
                    'iterations': [50, 100, 150, 200, 250]
                },

                "XGBoost": {
                    'learning_rate': [.01, 0.03, 0.05, 0.07, 0.09],
                    'n_estimators': [50, 100, 150, 200, 250]
                },

                "Lasso": {
                    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]
                },

                "Ridge": {
                    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]
                }
            }


            logging.info('Model evaluation started')
            report=self.evaluate_model(X_train, Y_train, X_test, Y_test, models, params)

            best_result=max(list(report.values()))
            best_model=list(report.keys())[list(report.values()).index(best_result)]

            logging.info(f"Best Model: {best_model} , Best Result: {best_result}")

            save_object(obj=models[best_model], file_path=self.model_trainer_config.model_path)
            logging.info('Model object saved as pickle file.')

            return best_model, best_result, report
            
        except Exception as e:
            raise CustomException(e, sys)