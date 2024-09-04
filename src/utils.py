import os
import dill
import sys
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Saves the given object to a file at the specified path using dill.
    """
    try:
        # Create the directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the object to a file in binary mode
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        # Raise a custom exception if something goes wrong
        raise CustomException(e, sys)
