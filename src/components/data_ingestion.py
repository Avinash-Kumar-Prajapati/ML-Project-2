import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import remove_outliers

@dataclass
class DataIngestionConfig:
    raw_data_path=os.path.join('artifacts', 'data.csv')
    train_data_path=os.path.join('artifacts', 'train.csv')
    test_data_path=os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info('Data ingestion is started')

            data=pd.read_csv('notebook/data/laptop_data.csv')

            df=remove_outliers(data)
            logging.info('Removed outliers from the data')

            df.drop_duplicates(inplace=True)
            logging.info('Removed duplicates from the data')

            os.makedirs('artifacts', exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)

            train_data, test_data=train_test_split(df, test_size=0.15, random_state=42)
            logging.info('Data is splitted into train and test data')

            train_data.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_data.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)
            logging.info('Data ingestion is completed')

        except Exception as e:
            raise CustomException(e, sys)