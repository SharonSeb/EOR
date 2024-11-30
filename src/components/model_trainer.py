import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import (AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor,)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pk1")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbours Regressor" : KNeighborsRegressor(),
                "XGB Classifier": XGBRegressor(),
                "AdaBoost Classifier" : AdaBoostRegressor(),
            }

            params = {
                "Random Forest": {
                    'max_depth':[int(x) for x in np.linspace(4, 16, num = 4)],
                    'n_estimators':[int(x) for x in np.linspace(start = 50, stop = 300, num = 5)],
                    'max_features': ['auto', 'sqrt'],
                    'min_samples_split': [2,5,10],
                    'min_samples_leaf':[1,2,4]
                    },
                "Decision Tree" :{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter' : ['best', 'random'],
                    'max_features': ['sqrt', 'log2'],
                    },
                "Gradient Boosting": {
                    'n_estimators': np.arange(50,200,25), 
                    'subsample':[0.7,0.8,0.9,1],
                    'max_features':[0.7,0.8,0.9,1],
                    'max_depth':[3,5,7,10],
                    },
                "Linear Regression": {},

                "K-Neighbours Regressor": {
                    'n_neighbors':[5,7,9,11],                
                    },

                "XGB Classifier": {
                    'n_estimators': [75,100,125,150], 
                    'subsample':[0.7, 0.8, 0.9, 1],
                    'gamma':[0, 1, 3, 5],
                    'colsample_bytree':[0.7, 0.8, 0.9, 1],
                    'colsample_bylevel':[0.7, 0.8, 0.9, 1]
                    },

                "AdaBoost Classifier": {
                    'n_estimators': [8,16,32,64,128,256], 
                    'learning_rate': [0.1, 0.5, 0.01],
                    }

            }

            model_report:dict = evaluate_models(X_train = X_train, y_train=y_train, X_test = X_test, y_test = y_test, models = models, param = params) 

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            #if best_model_score <0.6:
                #raise CustomException("No best model found")
            #logging.info(f"Best found model on both training and testing dataset")

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj = best_model)

            predicted = best_model.predict(X_test)

            r_score = r2_score(y_test, predicted)
            print(best_model)
            return r_score
        

        except Exception as e:
            raise CustomException(e,sys)
        
