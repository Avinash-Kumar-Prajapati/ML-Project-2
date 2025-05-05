import logging
import os
from datetime import datetime

LOG_FOLDER_NAME=f"{datetime.now().strftime("%d-%m-%Y")}"
LOG_FOLDER_PATH=os.path.join(os.getcwd(), 'logs', LOG_FOLDER_NAME)
os.makedirs(LOG_FOLDER_PATH, exist_ok=True)

LOG_FILE_NAME=f"{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}.log"
LOG_FILE_PATH=os.path.join(LOG_FOLDER_PATH, LOG_FILE_NAME)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)