import dill
import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import numpy as np

def SaveModel(filepath, obj):
    try:
        dir = os.path.dirname(filepath)

        os.makedirs(dir, exist_ok=True)

        with open(filepath, 'wb') as fileobj:
            dill.dump(obj, fileobj)
    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_model(file_path):
    try:
        with open(file_path, 'rb') as file_Obj:
            return dill.load(file_Obj)

    except Exception as e:
        logging.info("Exception occured in load the model")
        raise CustomException(e,sys)
    

def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        for i in len(models):
            report = {}

            model = list(models.values())[i]

            model.fit(x_train,y_train)

            y_pred_test = model.predict(x_test)

            test_r2_score = r2_score(y_test,y_pred_test)

            report[list(models.keys())[i]] == test_r2_score
        
        return report

    except Exception as e:
        logging.info("Exception occured in model training and evaluation")

def model_metrics(true,predicted):
    try:
        mae = mean_absolute_error(true,predicted)
        mse = mean_squared_error(true,predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(true,predicted)
        return mae,rmse,r2
    
    except Exception as e:
        logging.info("Exception occured in model evaluation")
        raise CustomException(e,sys)

def print_evaluated_results(xtrain,ytrain,xtest,ytest,model):
    try:
        ytrain_pred = model.predict(xtrain)
        ytest_pred = model.predict(xtest)

        model_train_mae , model_train_rmse, model_train_r2 = model_metrics(ytrain, ytrain_pred)
        model_test_mae , model_test_rmse, model_test_r2 = model_metrics(ytest, ytest_pred)

        print('Model performance for Training set')
        print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
        print("- R2 Score: {:.4f}".format(model_train_r2))

        print('----------------------------------')
    
        print('Model performance for Test set')
        print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
        print("- R2 Score: {:.4f}".format(model_test_r2))
    
    except Exception as e:
        logging.info('Exception occured during printing of evaluated results')
        raise CustomException(e,sys)