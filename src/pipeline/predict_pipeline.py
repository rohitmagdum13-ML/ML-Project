import sys
import os  # Import os for file path operations
import pandas as pd
from src.exception import CustomException  # Custom exception handling
from src.utils import load_object  # Utility to load saved objects (model, preprocessor)

class PredictPipeline:
    # Constructor for initializing the pipeline (currently does nothing)
    def __init__(self):
        pass

    # Method to make predictions based on input features
    def predict(self, features):
        try:
            # Define paths to the saved model and preprocessor artifacts
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            print("Before Loading")

            # Load the trained model and preprocessor objects
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            print("After Loading")

            # Preprocess the input features using the preprocessor
            data_scaled = preprocessor.transform(features)

            # Make predictions using the loaded model
            preds = model.predict(data_scaled)

            return preds
        
        except Exception as e:
            # Raise a custom exception in case of errors during prediction
            raise CustomException(e, sys)


class CustomData:
    # Constructor for receiving and initializing user input data
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str, 
                 lunch: str, test_preparation_course: str, reading_score: int, writing_score: int):
        
        # Assign the input data to the class attributes
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    # Method to convert the input data into a pandas DataFrame for model processing
    def get_data_as_data_frame(self):
        try:
            # Create a dictionary from the input data, with keys matching the model's expected features
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Convert the dictionary into a pandas DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            # Raise a custom exception in case of errors during data frame creation
            raise CustomException(e, sys)


