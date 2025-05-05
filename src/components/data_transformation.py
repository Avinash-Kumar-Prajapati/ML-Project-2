import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import TargetEncoder 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, parse_memory, remove_outliers, remove_word


@dataclass
class DataTransformationConfig:
    transformer_path=os.path.join('artifacts', 'transformer.pkl')


class CustomDataTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            # Convert column headers to lower case
            X.columns=X.columns.str.lower()

            # Extarction of screen resolution values and creation of a new features
            X['touchscreen']=(X['screenresolution'].apply(lambda x:1 if 'touchscreen' in x.lower() else 0))
            X['ips_panel']=(X['screenresolution'].apply(lambda x:1 if 'ips panel' in x.lower() else 0))

            X['screenresolution']=(X['screenresolution'].apply(remove_word))

            X['screen_type']=(X['screenresolution'].apply(lambda x:''.join(x.split(' ')[0:-1])))

            X['screenresolution']=(X['screenresolution'].apply(lambda x:x.split(' ')[-1]))
            X['screen_width']=X['screenresolution'].apply(lambda x:(int(x.split('x')[0])))
            X['screen_height']=X['screenresolution'].apply(lambda x:(int(x.split('x')[1])))
            X.drop('screenresolution', axis=1, inplace=True)

            # Extraction of SSD, HDD, Flash and Hybrid storage from the Memory field values
            X[['ssd','hdd','flash','hybrid']]=X['memory'].apply(lambda x: pd.Series(parse_memory(x)))
            X['total_memory']=X['ssd']+X['hdd']+X['flash']+X['hybrid']
            X.drop('memory', axis=1, inplace=True)

            # Extraction of Weight value nos. and transforming it to float
            X['weight']=X['weight'].apply(lambda x:x.replace('kg',''))
            X['weight']=X['weight'].astype(float)

            # Extraction of RAM value nos. and transforming it to float
            X['ram']=X['ram'].apply(lambda x:x.replace('GB',''))
            X['ram']=X['ram'].astype(int)

            return X
        
        except Exception as e:
            raise CustomException(e, sys)
        
        
class TargetEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = TargetEncoder()
    
    def fit(self, X, y):
        self.encoder.fit(X, y)
        return self
    
    def transform(self, X):
        return self.encoder.transform(X)


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:            
            cat_features_1= ['company', 'typename', 'opsys', 'touchscreen', 'ips_panel', 'screen_type']
            cat_features_2= ['cpu', 'gpu']
            num_features= ['inches', 'ram', 'weight', 'screen_width', 'screen_height', 'ssd', 'hdd',
                          'flash', 'hybrid', 'total_memory']

            num_pipeline=Pipeline(
                [
                    ('standard scaler', StandardScaler())
                ]
            )

            cat_pipeline_1=Pipeline(
                [
                    ('encoder', OneHotEncoder(handle_unknown='ignore')),
                ]
            )

            cat_pipeline_2=Pipeline(
                [
                    ('encoder', TargetEncoderWrapper()),
                    ('standard scaler', StandardScaler())
                ]
            )

            transformer=Pipeline(
                [
                    ('data_transform', CustomDataTransformer()),
                    ('column_transform', ColumnTransformer([('num_pipeline', num_pipeline, num_features),
                                                            ('cat_pipeline1', cat_pipeline_1, cat_features_1),
                                                            ('cat_pipeline2', cat_pipeline_2, cat_features_2)]))
                ]
            )

            return transformer

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)
            logging.info('Read Train data and Test data from csv file')

            target='Price'

            input_train_data=train_data.drop(target, axis=1)
            target_train_data=train_data[target]

            input_test_data=test_data.drop(target, axis=1)
            target_test_data=test_data[target]

            transformer_obj=self.get_data_transformer_obj()

            logging.info('Loaded transformer object')

            input_train_arr=transformer_obj.fit_transform(input_train_data, target_train_data)
            input_test_arr=transformer_obj.transform(input_test_data)
           
            train_data_transformed=np.c_[input_train_arr, np.array(target_train_data)]
            test_data_transformed=np.c_[input_test_arr, np.array(target_test_data)]

            logging.info('Data transformation completed')

            save_object(obj=transformer_obj, file_path=self.data_transformation_config.transformer_path)
            logging.info('Transformer object saved as pickle file.')


            return train_data_transformed, test_data_transformed

        except Exception as e:
            raise CustomException(e, sys)

            
            