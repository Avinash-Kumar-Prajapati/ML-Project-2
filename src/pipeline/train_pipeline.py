from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import warnings

if __name__=='__main__':

    warnings.filterwarnings("ignore", category=UserWarning)

    data_ingestion=DataIngestion()
    data_ingestion.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_data, test_data=data_transformation.initiate_data_transformation(train_path=DataIngestionConfig.train_data_path, 
                                                                           test_path=DataIngestionConfig.test_data_path)
    model_trainer=ModelTrainer()
    model, score, report=model_trainer.initiate_model_trainer(train_data=train_data, test_data=test_data)

    print("\n============================\n")
    print(f"Model: {model} \nR2_Score: {score}")
    print(f"\n{report}")