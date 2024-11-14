# tous qui reading the data
import os
import sys
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer 
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset
            df = pd.read_csv('notebook\data\data_v1.csv')
            logging.info('Read the dataset as dataframe')
            logging.info(f"Droppindf.shape: {df.shape}")

            df.shape
            # Drop unnecessary columns
            columns_to_drop = ['id', 'cik', 'ticker', 'accessionNo', 'companyName', 'fy', 'fp', 'form', 'filed' ]
            existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
            logging.info(f"Dropping columns: {existing_columns_to_drop}")
            df = df.drop(columns=existing_columns_to_drop, errors='ignore')
            logging.info(f"Droppindf.shape: {df.shape}")

            # Remove duplicate rows
            num_duplicates = df.duplicated().sum()
            logging.info(f"Number of duplicate rows before dropping duplicates: {num_duplicates}")
            df = df.drop_duplicates()
            logging.info("Duplicates removed from the dataset")

            columns_to_drop1 = ['Current_Other_Assets', 'Nonoperating_Income', 'Intangible_Assets', 'GrossProfit']
            existing_columns_to_drop = [col for col in columns_to_drop1 if col in df.columns]
            logging.info(f"Dropping columns: {existing_columns_to_drop}")
            df = df.drop(columns=existing_columns_to_drop, errors='ignore')

            
            # Create directories and save raw data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Split the data into training and testing sets
            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save training and testing datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    
    data_transformer = DataTransformation()
    X_train, y_train, X_test, y_test=data_transformer.initiate_data_transformation(train_data,test_data)
    print(f"starting moel trainer")

    modeltrainer = ModelTrainer()
    result = modeltrainer.initiate_model_trainer(X_train, y_train, X_test, y_test)
    print("Model Training Result:", result)