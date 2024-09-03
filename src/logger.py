import os
import sys
import logging

# Define the logging format
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Directory where log files will be stored
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")

# Create the log directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Configure the logging
logging.basicConfig(
    level=logging.INFO,                  # Set the logging level to INFO
    format=logging_str,                  # Use the defined format for log messages

    handlers=[
        logging.FileHandler(log_filepath),  # Write log messages to a file
        logging.StreamHandler(sys.stdout)   # Also print log messages to the console (stdout)
    ]
)

# Create a logger object for your project
logger = logging.getLogger("MLProject")

# Example usage of logger
logger.info("Logging setup is complete!")
