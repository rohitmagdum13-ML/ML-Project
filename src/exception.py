import sys
import logging

# Function to extract error details
def error_message_detail(error, error_detail: sys):
    # Get traceback information
    _, _, exc_tb = error_detail.exc_info()
    # Get the filename and line number where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    # Format the error message with script name, line number, and error message
    error_message = f"Error occurred in script: [{file_name}] at line [{exc_tb.tb_lineno}] - Error message: [{str(error)}]"
    return error_message

# Custom exception class to handle errors
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        # Use the error_message_detail function to format the error message
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
