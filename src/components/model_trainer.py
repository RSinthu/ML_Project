from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import os
import sys
from src.utils import save_model,model_metrics
from datetime import datetime

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts','model.pkl')
    model_report_path = os.path.join('artifacts','model_evaluation_report.txt')

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
                "Random_Forest_Regressor":RandomForestRegressor(),
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

            model_report = {}
            best_models = {}
            all_model_results = {}
            
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
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    
                else:  # For models without hyperparameters (Linear Regression)
                    best_model = models[model_name]
                    best_model.fit(x_train, y_train)
                    best_params = None

                y_train_pred = best_model.predict(x_train)
                y_test_pred = best_model.predict(x_test)

                train_mae, train_rmse, train_r2 = model_metrics(y_train, y_train_pred)
                test_mae, test_rmse, test_r2 = model_metrics(y_test,y_test_pred)

                all_model_results[model_name] = {
                    'best_parameters': best_params,
                    'train_metrics':{
                        'r2_score':float(train_r2),
                        'mae':float(train_mae),
                        'rmse':float(train_rmse)
                    },
                    'test_metrics':{
                        'r2_score': float(test_r2),
                        'mae': float(test_mae),
                        'rmse':float(test_rmse)
                    }
                }

                model_report[model_name] = test_r2
                best_models[model_name] = best_model

                logging.info(f"{model_name} - Test R2 Score: {test_r2}")

            # Select the best model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = best_models[best_model_name]
            
            logging.info(f"Best Model: {best_model_name} with R2 Score: {best_model_score}")

            self.create_consolidated_report(all_model_results, best_model_name, best_model_score)
            
            # Save the best model
            save_model(
                self.model_trainer_config.trained_model_path,
                best_model
            )
            

        except Exception as e:
            logging.info("Error occured in model training")
            raise CustomException(e,sys)
    
    def create_text_report(self, report_data):
        try:
            with open(self.model_trainer_config.model_report_path, 'w') as f:
                f.write("MODEL EVALUATION COMPREHENSIVE REPORT\n")
                f.write("=" * 60 + "\n\n")

                f.write(f"Evaluation Date: {report_data['evaluation_timestamp']}\n")
                f.write(f"Total Models Evaluated: {report_data['total_models_evaluated']}\n\n")

                f.write("SUMMARY STATISTICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Best Model: {report_data['best_model']['name']}\n")
                f.write(f"Best R² Score: {report_data['best_model']['r2_score']:.4f}\n")

                f.write("MODEL RANKINGS (By R² Score)\n")
                f.write("-" * 35 + "\n")
                sorted_models = sorted(report_data['all_models'].items(), 
                                     key=lambda x: x[1]['test_metrics']['r2_score'], 
                                     reverse=True)
                
                for i, (model_name, model_data) in enumerate(sorted_models, 1):
                    f.write(f"{i:2d}. {model_name:<25} R²: {model_data['test_metrics']['r2_score']:.4f}\n")
                
                f.write("\n" + "=" * 60 + "\n\n")
                
                f.write("DETAILED MODEL PERFORMANCE\n")
                f.write("=" * 60 + "\n\n")
                
                for model_name, model_data in sorted_models:
                    f.write(f"MODEL: {model_name}\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"Best Parameters: {model_data.get('best_parameters', 'No hyperparameter tuning')}\n\n")
                    
                    train_metrics = model_data['train_metrics']
                    f.write("Training Performance:\n")
                    f.write(f"  R² Score: {train_metrics['r2_score']:.4f}\n")
                    f.write(f"  MAE:      {train_metrics['mae']:.4f}\n")
                    f.write(f"  RMSE:     {train_metrics['rmse']:.4f}\n\n")

                    test_metrics = model_data['test_metrics']
                    f.write("Test Performance:\n")
                    f.write(f"  R² Score: {test_metrics['r2_score']:.4f}\n")
                    f.write(f"  MAE:      {test_metrics['mae']:.4f}\n")
                    f.write(f"  RMSE:     {test_metrics['rmse']:.4f}\n\n")
                    
                    f.write("Overfitting Analysis:\n")
                    r2_diff = train_metrics['r2_score'] - test_metrics['r2_score']
                    f.write(f"  R² Difference: {r2_diff:.4f}\n")
                    f.write(f"  Overfitting: {'Yes' if r2_diff > 0.1 else 'No'}\n\n")
                    f.write("-" * 50 + "\n\n")


        except Exception as e:
            logging.info("Exception occured in create text report")
            raise CustomException(e,sys)
        

    def create_consolidated_report(self, all_model_results, best_model_name, best_model_score):
        try:
            report_data = {
                "evaluation_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "total_models_evaluated": len(all_model_results),
                "best_model": {
                    "name": best_model_name,
                    "r2_score": float(best_model_score)
                },
                "all_models": all_model_results
            }

            self.create_text_report(report_data)

            logging.info("Consolidated model evaluation report created successfully")
        except Exception as e:
            logging.info("Exception occured at create the consolidated report")
            raise CustomException(e,sys)

