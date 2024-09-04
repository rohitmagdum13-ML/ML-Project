import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

# Configuration for data ingestion paths
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

# Class for handling data ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

     # Method to initiate data ingestion
    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion...")
        try:
            # Path to the dataset
            data_path = 'notebooks\data\stud.csv'
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at {data_path}")

            # Read the dataset into a DataFrame
            df = pd.read_csv(data_path)
            logging.info('Dataset successfully read into DataFrame')

            # Create necessary directories for storing artifacts
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info('Raw data saved')

            # Split data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train-test split completed")

            # Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path)
            test_set.to_csv(self.ingestion_config.test_data_path)
            logging.info("Data ingestion completed successfully")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
           
        except Exception as e:
            raise CustomException(e, sys)
        
# Main execution
if __name__ == "__main__":
    # Instantiate DataIngestion class and perform ingestion
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()

