import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

project_name = "ML_Project"

# List of files to create
list_of_files = [
    "src/__init__.py",
    "src/exception.py",
    "src/logger.py",
    "src/utils.py",
    
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    
    "src/pipeline/__init__.py",
    "src/pipeline/predict_pipeline.py",
    "src/pipeline/train_pipeline.py",
    
    "notebooks/main.ipynb",
    "notebooks/data/abc.csv",
    
    "requirements.txt",
    "setup.py",
]

# Iterate over the list of files
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # Create directories if they don't exist
    if filedir:
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file {filename}")

    # Create the file if it doesn't exist or is empty
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "w") as f:
            pass  # Create an empty file
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"File {filename} already exists")
