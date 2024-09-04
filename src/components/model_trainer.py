import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

# Configuration class for saving the trained model
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

# Class for model training
class ModelTrainer:
    def __init__(self):
        self.model_trainer_congfig = ModelTrainerConfig()

    # Main function to initiate the model training process
    def initiate_model_trainer(self, train_array, test_array):
        try:
            # Split training and test data into features (X) and target (y)
            logging.info("Splitting training and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], # All columns except the last one (features)
                train_array[:, -1],  # The last column (target)
                test_array[:, :-1],  # Same for test set
                test_array[:, -1]
            )


            # Define models to evaluate
            logging.info("Initializing models for evaluation")
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }


            # Define hyperparameters for each model
            logging.info("Setting hyperparameters for each model")
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 10, 20, 30],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_features': ['sqrt', 'log2', None],
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [3, 4, 5, 6],
                },
                "Linear Regression": {},  # No hyperparameters for Linear Regression
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [3, 4, 5, 6],
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                }
            }

            # Evaluate models and get their scores
            logging.info("Evaluating models with cross-validation")
            model_report: dict = evaluate_models(
                X_train = X_train,
                y_train = y_train, 
                X_test = X_test,
                y_test = y_test,
                models = models,
                param = params
            )

            logging.info(f"Model evaluation completed. Results: {model_report}")


            # Find the best model and its score
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            


            # If no model performs well, raise an exception
            if best_model_score < 0.6:
                logging.warning("No model found with an acceptable score. Raising exception.")
                raise CustomException("No suitable model found with acceptable performance")

            logging.info(f"Best model found: {best_model_name}")

            # Save the best model
            save_object(
                file_path = self.model_trainer_congfig.trained_model_file_path,
                obj = best_model
            )


            # Make predictions and calculate R2 score
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square
            
        except Exception as e:
            raise CustomException(e, sys)

    