import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# Configuration class for saving the preprocessor object path
@dataclass
class DataTransformationCongfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

# Class to handle data transformation
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationCongfig()

    # Function to create and return the data transformer pipeline
    def get_data_transformer(self):
        '''
        This function is responsible for creating and returning a data transformation pipeline.
        '''
        try:
            # Define numerical and categorical columns
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Pipeline for numerical features
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")), # Fill missing values
                    ("scaler", StandardScaler()) # Standardize numerical features
                ]
            )

            # Pipeline for categorical features
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")), # Fill missing values
                    ("one_hot_encoder", OneHotEncoder()), # One-hot encode categorical features
                    ("scaler", StandardScaler(with_mean=False)) # Standardize the one-hot encoded features
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine both pipelines into a single preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", numerical_pipeline, numerical_columns),
                    ("cat_pipeline", categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    # Function to initiate data transformation
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Loaded train and test datasets.")

            # Get the preprocessing pipeline
            preprocessing_obj = self.get_data_transformer()

            # Define target column and numerical columns
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Separate input features and target variable for training and testing
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing on train and test datasets.")

            # Apply transformations on the train and test datasets
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine the transformed input features with target variable
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object.")

            # Save the preprocessing object
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            # Return transformed train, test data and the preprocessor file path
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
                    raise CustomException(e, sys)
