from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_model
import os
import pandas as pd
import numpy as np
import sys

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    raw_data_path = os.path.join('artifacts','data.csv')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            df = pd.read_csv(self.data_transformation_config.raw_data_path)

            exclude_columns = ['id','price']
            feature_columns = [col for col in df.columns if col not in exclude_columns]

            numerical_features = []
            categorical_features = []

            for feature in feature_columns:
                if df[feature].dtype in ['int64','float64']:
                    numerical_features.append(feature)
                else:
                    categorical_features.append(feature)
            
            logging.info(f"Numerical features:{numerical_features}")
            logging.info(f"Categorical features:{categorical_features}")

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="median")),
                    ('scaler', StandardScaler())
                ]
            )

            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="most_frequent")),
                    ('ordinal_encoder', OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories]))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical_pipeline',num_pipeline,numerical_features),
                    ('categorical_pipeline',cat_pipeline,categorical_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            logging.info("Error occured in Data Transformation")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train and Test initial data completed")

            target_column = 'price'
            columns_to_drop = [target_column,'id']

            preprocessing_obj = self.get_data_transformation_obj()

            input_feature_train_df = train_df.drop(columns=columns_to_drop,axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=columns_to_drop,axis=1)
            target_feature_test_df = test_df[target_column]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_model(self.data_transformation_config.preprocessor_obj_file_path, preprocessing_obj)

            return (
                train_arr,
                test_arr
            )

        except Exception as e:
            logging.info("Error occured in initiate data preprocessing")
            raise CustomException(e,sys)
 
            
