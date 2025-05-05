import pandas as pd
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.utils import load_object

class CustomData:
    def __init__(self, Company, TypeName, Inches, ScreenResolution, CPU, RAM, Memory, GPU, OpSys, Weight):
        self.Company=Company
        self.TypeName=TypeName
        self.Inches=Inches
        self.ScreenResolution=ScreenResolution
        self.CPU=CPU
        self.RAM=RAM
        self.Memory=Memory
        self.GPU=GPU
        self.OpSys=OpSys
        self.Weight=Weight

    def get_data_as_dataframe(self):
        try:
            data={
                'Company': [self.Company],
                'TypeName': [self.TypeName],
                'Inches': [self.Inches],
                'ScreenResolution': [self.ScreenResolution],
                'CPU': [self.CPU],
                'RAM': [self.RAM],
                'Memory': [self.Memory],
                'GPU': [self.GPU],
                'OpSys': [self.OpSys],
                'Weight': [self.Weight]
                }
            
            data_df=pd.DataFrame(data)

            return data_df
        
        except Exception as e:
            raise CustomException(e, sys)
    
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, data):
        try:
            transformer_path=DataTransformationConfig.transformer_path
            model_path=ModelTrainerConfig.model_path

            transformer=load_object(transformer_path)
            model=load_object(model_path)

            transformed_data=transformer.transform(data)
            prediction=model.predict(transformed_data)

            return prediction
        
        except Exception as e:
            raise CustomException(e, sys)
