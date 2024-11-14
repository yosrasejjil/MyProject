from sklearn.model_selection import train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import RobustScaler
from src.logger import logging
from src.exception import CustomException
import os
import sys
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # Enables the IterativeImputer feature
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from dataclasses import dataclass
from src.utils import save_object
from imblearn.over_sampling import ADASYN
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours
from collections import Counter
import pandas as pd
@dataclass
class DataTransformationConfig:
    cleaning_pipeline_path = os.path.join("artifacts", "cleaning_pipeline.pkl")
    missing_path = os.path.join('artifacts', "preprocessor.pkl")
    scaling_pipeline_path = os.path.join('artifacts', "scaling_pipeline.pkl")

    feature_engineering_pipeline_path = os.path.join('artifacts', "feature_engineering.pkl")
    feature_selec_obj_path = os.path.join('artifacts', "feature_selection.pkl")
    

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig() 

    def create_cleaning_pipeline(self, missing_value_threshold=55.0):
        try:
            # Function to drop columns only if they exist in the DataFrame
            def drop_specified_columns(X, columns_to_drop):
                # Filter the columns to drop based on those present in the DataFrame
                existing_columns_to_drop = [col for col in columns_to_drop if col in X.columns]
                return X.drop(existing_columns_to_drop, axis=1, errors='ignore')
        
            # Pipeline for cleaning and reducing data
            cleaning_pipeline = Pipeline(steps=[
                ('drop_specified_columns', FunctionTransformer(
                    drop_specified_columns, 
                    kw_args={'columns_to_drop': [
                        'id', 'cik', 'ticker', 'accessionNo', 'companyName', 'fy', 'fp', 'form', 'filed' ,  'Current_Other_Assets', 'Nonoperating_Income', 'Intangible_Assets', 'GrossProfit'
                    ]})
                ),
                # Uncomment and adjust the following line if you need to drop columns based on missing values
                # ('drop_missing_value_columns', FunctionTransformer(self.drop_missing_value_columns, kw_args={'missing_value_threshold': missing_value_threshold}))
            ])
        
            return cleaning_pipeline
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def calculate_ratios(data):
        """Calculate financial ratios as in the notebook."""
        ratios = pd.DataFrame()

        ratios['R1'] = data['ShortTerm_Debt'] / data['Stockholder_Equity']  # Short Term Debt / Equity
        ratios['R2'] = data['Liabilities'] / data['Assets']  # Liabilities / Total Assets
        ratios['R3'] = data['Stockholder_Equity'] / data['Assets']  # Total Equity / Total Assets
        ratios['R4'] = data['Assets'] / data['Stockholder_Equity']  # Total Assets / Total Equity
        ratios['R5'] = data['Cash'] / data['Assets']  # Cash / Total Assets
        ratios['R6'] = data['Working_capital'] / data['Assets']  # Working Capital / Total Assets
        ratios['R7'] = data['Current_Assets'] / data['Current_liabilities']  # Current Ratio
        ratios['R8'] = data['NetIncome'] / data['Assets']  # Net Income / Total Assets (ROA)
        ratios['R9'] = data['Earning_Before_Interest_And_Taxes'] / data['InterestExpense']  # EBIT / Interest Expenses
        ratios['R10'] = data['LongTerm_Debt'] / data['Assets']  # Long-term Debt / Total Assets
        ratios['R11'] = (data['ShortTerm_Debt'] + data['LongTerm_Debt']) / data['Assets']  # Debt Dependency Ratio
        ratios['R12'] = (data['ShortTerm_Debt'] + data['LongTerm_Debt']) / (data['Cash'] + data['Assets'])  # Debt Capacity Ratio
        ratios['R13'] = (data['ShortTerm_Debt'] + data['LongTerm_Debt']) / data['Revenues']  # Debt / Total Revenue
        ratios['R14'] = data['AccountsReceivable'] / data['Liabilities']  # Accounts Receivable / Liabilities

        # Auxiliary variables for cash flows
        ratios['AV1'] = data['Cash']
        ratios['AV2'] = data['NetCash_OperatingActivities']
        ratios['AV3'] = data['NetCash_InvestingActivities']
        ratios['AV4'] = data['NetCash_FinancingActivities']

        return ratios

    def create_missing_value_pipeline(self):
        """Create a pipeline for handling missing values."""
        try:
            def random_fill(X):
                """
                Randomly fills missing data by sampling from existing non-missing values.
                """
                return X.apply(lambda col: col.fillna(np.random.choice(col.dropna())) if col.isnull().sum() > 0 else col, axis=0)

            # Wrap the `random_fill` function into a compatible transformer
            random_fill_transformer = FunctionTransformer(random_fill, validate=False)

            # Define the pipeline
            pipeline = Pipeline(steps=[
                ('random_fill', random_fill_transformer),  # Step 1: Random filling of missing values
                ('iterative_imputer', IterativeImputer(
                    estimator=LinearRegression(),
                    max_iter=10,
                    random_state=0
                ))  # Step 2: Iterative imputation (MICE)
            ])
            return pipeline
        except Exception as e:
            raise CustomException(f"Error in create_missing_value_pipeline: {e}")

    def create_scaling_pipeline(self):
        """Create a scaling pipeline."""
        try:
            pipeline = Pipeline(steps=[
                ('scaler', RobustScaler())
            ])
            return pipeline
        except Exception as e:
            raise CustomException(f"Error in create_scaling_pipeline: {e}")
    def feature_selection_function(self):
            try:
            # Function to drop specified columns
                def drop_specified_features(X):
                    # Predefined features to drop
                    features_to_drop =['R11', 'R1', 'R9', 'R8', 'R10']
                    #features_to_drop = ['R1', 'R13', 'R11', 'R12', 'R10', 'R8', 'R2']
                    logging.info(f"Dropping the predefined features: {features_to_drop}")
                    return X.drop(features_to_drop, axis=1, errors='ignore')
                
                # Create the feature selection pipeline
                feature_selection_pipeline = Pipeline(steps=[
                    ('drop_specified_features', FunctionTransformer(drop_specified_features))
                ])
                
                return feature_selection_pipeline
            except Exception as e:
                logging.error(f"Error while creating feature selection pipeline: {str(e)}")
                raise e
            

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info(f"Loading train and test datasets from {train_path} and {test_path}")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = 'is_bankrupt'
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # Separate majority and minority classes
            X_train_majority = X_train[y_train == 0].dropna()
            y_train_majority = y_train[X_train_majority.index]
            X_train_minority = X_train[y_train == 1]
            y_train_minority = y_train[y_train == 1]

            X_test_majority = X_test[y_test == 0].dropna()
            y_test_majority = y_test[X_test_majority.index]
            X_test_minority = X_test[y_test == 1]
            y_test_minority = y_test[y_test == 1]

            # Impute minority class data
            missing_pipeline = self.create_missing_value_pipeline()
            X_train_minority_imputed = pd.DataFrame(
                missing_pipeline.fit_transform(X_train_minority),
                columns=X_train_minority.columns
            )
            X_test_minority_imputed = pd.DataFrame(
                missing_pipeline.transform(X_test_minority),
                columns=X_test_minority.columns
            )

            # Combine data
            X_train = pd.concat([X_train_majority, X_train_minority_imputed], axis=0).reset_index(drop=True)
            y_train = pd.concat([y_train_majority, y_train_minority], axis=0).reset_index(drop=True)
            X_test = pd.concat([X_test_majority, X_test_minority_imputed], axis=0).reset_index(drop=True)
            y_test = pd.concat([y_test_majority, y_test_minority], axis=0).reset_index(drop=True)

            # Scaling
            scaling_pipeline = self.create_scaling_pipeline()
            X_train_scaled = scaling_pipeline.fit_transform(X_train)
            X_test_scaled = scaling_pipeline.transform(X_test)

            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

            # Feature engineering
            X_train_ratios = self.calculate_ratios(X_train_scaled)
            X_test_ratios = self.calculate_ratios(X_test_scaled)

            X_train = pd.concat([X_train_scaled, X_train_ratios], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            X_test = pd.concat([X_test_scaled, X_test_ratios], axis=1).replace([np.inf, -np.inf], np.nan).dropna()

            y_train = y_train[X_train.index]
            y_test = y_test[X_test.index]

            logging.info(f"Transformed train shape: {X_train.shape}, Transformed test shape: {X_test.shape}")

                        # Save pipelines

            input_feature_train_df = X_train
            input_feature_test_df = X_test      
            # Apply the feature selection function
            # Create the feature selection pipeline
            logging.info("Creating the feature selection pipeline.")
            feature_selection_pipeline = self.feature_selection_function()

            # Apply the pipeline to the training and test data
            logging.info("Fitting and transforming the feature selection pipeline on training data.")
            input_feature_train_selected = feature_selection_pipeline.fit_transform(input_feature_train_df)

            logging.info("Transforming the test data with the fitted feature selection pipeline.")
            input_feature_test_selected = feature_selection_pipeline.transform(input_feature_test_df)

            # Save the pipeline
            logging.info("Saving the feature selection pipeline.")
            # Save pipelines
            save_object(self.data_transformation_config.missing_path, missing_pipeline)
            save_object(self.data_transformation_config.scaling_pipeline_path, scaling_pipeline)
            save_object(self.data_transformation_config.feature_selec_obj_path, feature_selection_pipeline)

            # Log the final shapes of the transformed data
            logging.info(f"Input features shape (train) after feature selection: {input_feature_train_selected.shape}")
            logging.info(f"Input features shape (test) after feature selection: {input_feature_test_selected.shape}")


            # For example, applying ADASYN to handle any remaining imbalance in the training set
            adasyn = ADASYN(random_state=42)
            X_train, y_train = adasyn.fit_resample(input_feature_train_selected, y_train)

            # Check the new class distribution after applying ADASYN
            logging.info(f"Class distribution after ADASYN: {y_train.value_counts()}")
            
            # Step 18: Return the transformed datasets
            return X_train, y_train,input_feature_test_selected,y_test
            
        # Return the final balanced datasets
        except Exception as e:
            raise CustomException(e, sys)



    ### Drop specified columns function
    @staticmethod
    def drop_specified_columns(X, columns_to_drop):
        return X.drop(columns=columns_to_drop, axis=1)

    ### Drop missing value columns function
    @staticmethod
    def drop_missing_value_columns(X, missing_value_threshold):
        missing_percentage = (X.isnull().sum() / len(X)) * 100
        columns_to_drop = missing_percentage[missing_percentage > missing_value_threshold].index.tolist()
        return X.drop(columns=columns_to_drop, axis=1)
