from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import os
import sys
from src.utils import evaluate_model,SaveModel
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting dependent and independent columns from the train and test array")
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Linear_Regression":LinearRegression(),
                "Lasso":Lasso(),
                "Ridge":Ridge(),
                "Random_Forest_Regresssor":RandomForestRegressor(),
                "Decision_Tree_Regressor":DecisionTreeRegressor(),
                "XGBoost":XGBRegressor()
            }

            params = {
                "Linear_Regression": {},
                "Lasso": {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                },
                "Ridge": {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                },
                "Random_Forest_Regressor": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                "Decision_Tree_Regressor": {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "XGBoost": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            }

            # Perform GridSearchCV for each model
            model_report = {}
            best_models = {}
            
            for model_name in models:
                logging.info(f"Training {model_name} with GridSearchCV")
                
                if params[model_name]:  # If hyperparameters exist
                    grid_search = GridSearchCV(
                        estimator=models[model_name],
                        param_grid=params[model_name],
                        cv=5,  # 5-fold cross-validation
                        scoring='r2',
                        n_jobs=-1,
                        verbose=1
                    )
                    grid_search.fit(x_train, y_train)
                    best_models[model_name] = grid_search.best_estimator_
                    
                    # Evaluate on test set
                    y_pred = grid_search.best_estimator_.predict(x_test)
                    test_score = r2_score(y_test, y_pred)
                    model_report[model_name] = test_score
                    
                    logging.info(f"{model_name} - Best params: {grid_search.best_params_}")
                    logging.info(f"{model_name} - Test R2 Score: {test_score}")
                    
                else:  # For models without hyperparameters (Linear Regression)
                    model = models[model_name]
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    test_score = r2_score(y_test, y_pred)
                    model_report[model_name] = test_score
                    best_models[model_name] = model

            # Select the best model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = best_models[best_model_name]
            
            logging.info(f"Best Model: {best_model_name} with R2 Score: {best_model_score}")
            
            # Save the best model
            SaveModel(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )
            
            return best_model_score

        except Exception as e:
            logging.info("Error occured in model training")
            raise CustomException(e,sys)

