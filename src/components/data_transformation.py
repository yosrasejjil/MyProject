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

@dataclass
class DataTransformationConfig:
    cleaning_pipeline_path = os.path.join("artifacts", "cleaning_pipeline.pkl")
    missing_path = os.path.join('artifacts', "preprocessor.pkl")
    scaling_pipeline_path = os.path.join('artifacts', "scaling_pipeline.pkl")

    feature_engineering_pipeline_path = os.path.join('artifacts', "feature_engineering.pkl")
    feature_selec_obj_path = os.path.join('artifacts', "feature_selection.pkl")
    # Paths for feature selection techniques
    """ pearson_selection_path: str = os.path.join('artifacts', "feature_selection_pearson.pkl")
    spearman_selection_path: str = os.path.join('artifacts', "feature_selection_spearman.pkl")
    mutual_info_selection_path: str = os.path.join('artifacts', "feature_selection_mutual_info.pkl")
    variance_selection_path: str = os.path.join('artifacts', "feature_selection_variance.pkl")
    random_forest_rfe_selection_path: str = os.path.join('artifacts', "feature_selection_random_forest_rfe.pkl") """


class DataTransformation:
    def __init__(self, threshold=0.01, method='variance', k_best=5, n_features_to_select=5, random_state=42):
        self.data_transformation_config = DataTransformationConfig()
        self.threshold = threshold
        self.method = method
        self.k_best = k_best
        self.n_features_to_select = n_features_to_select
        self.random_state = random_state
        logging.info(f"Data transformation initialized with feature selection method: {self.method}")

    ### Feature Selection

    ### Pearson Correlation Feature Selection Function
    def pearson_correlation_pipeline(self, X, y):
        try:
            logging.info(f"Creating Pearson correlation feature selection pipeline with threshold = {self.threshold}")

            def select_pearson_correlation(X, y, threshold):
                corr_matrix = X.corr(method='pearson')
                correlated_groups = []
                target_corr = X.apply(lambda col: np.corrcoef(col, y)[0, 1])

                # Find highly correlated groups
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > threshold:
                            correlated_groups.append((corr_matrix.columns[i], corr_matrix.columns[j]))

                # Select features to drop based on correlation with target
                features_to_drop = []
                for feature1, feature2 in correlated_groups:
                    if abs(target_corr[feature1]) > abs(target_corr[feature2]):
                        features_to_drop.append(feature2)
                    else:
                        features_to_drop.append(feature1)

                return X.drop(columns=set(features_to_drop), axis=1)

            # Pipeline for Pearson correlation feature selection
            pipeline = Pipeline(steps=[
                ('pearson_corr', FunctionTransformer(lambda X: select_pearson_correlation(X, y, self.threshold), validate=False))
            ])
            return pipeline
        except Exception as e:
            raise CustomException(e, sys)

    ### Spearman Correlation Feature Selection Function
    def spearman_correlation_pipeline(self, X, y):
        try:
            logging.info(f"Creating Spearman correlation feature selection pipeline with threshold = {self.threshold}")

            def select_spearman_correlation(X, y, threshold):
                corr_matrix = X.corr(method='spearman')
                correlated_groups = []
                target_corr = X.apply(lambda col: np.corrcoef(col, y)[0, 1])

                # Find highly correlated groups
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > threshold:
                            correlated_groups.append((corr_matrix.columns[i], corr_matrix.columns[j]))

                # Select features to drop based on correlation with target
                features_to_drop = []
                for feature1, feature2 in correlated_groups:
                    if abs(target_corr[feature1]) > abs(target_corr[feature2]):
                        features_to_drop.append(feature2)
                    else:
                        features_to_drop.append(feature1)

                return X.drop(columns=set(features_to_drop), axis=1)

            # Pipeline for Spearman correlation feature selection
            pipeline = Pipeline(steps=[
                ('spearman_corr', FunctionTransformer(lambda X: select_spearman_correlation(X, y, self.threshold), validate=False))
            ])
            return pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def mutual_information_pipeline(self, y):
        try:
            logging.info("Creating Mutual Information feature selection pipeline")
            pipeline = Pipeline(steps=[
                ('mutual_info', SelectKBest(mutual_info_classif, k=self.k_best))
            ])
            return pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def random_forest_rfe_pipeline(self, y):
        try:
            logging.info("Creating RFE with Random Forest pipeline")
            rf_model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            pipeline = Pipeline(steps=[
                ('rfe', RFE(estimator=rf_model, n_features_to_select=self.n_features_to_select))
            ])
            return pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def variance_and_drop_pipeline(self, columns_to_drop):
        """
        Creates a pipeline that applies variance thresholding and drops specified columns.

        Parameters:
        - columns_to_drop: List of columns to be dropped after applying the variance threshold.

        Returns:
        - A scikit-learn Pipeline object that applies variance threshold and drops specified columns.
        """
        try:
            # Step 1: Create the variance threshold pipeline
            logging.info(f"Creating Variance Threshold pipeline with threshold = {self.threshold}")
            variance_pipeline = Pipeline(steps=[
                ('variance_threshold', VarianceThreshold(threshold=self.threshold))
            ])

            # Step 2: Define the final pipeline that includes the variance threshold and column dropping
            final_pipeline = Pipeline(steps=[
                ('variance', variance_pipeline),
                ('drop_columns', FunctionTransformer(lambda X: self.drop_specified_columns(X, columns_to_drop), validate=False))
            ])

            return final_pipeline
        except Exception as e:
            raise CustomException(e, sys)

    ### Features Engineering
    def financial_ratios_pipeline(self):
        try:
            logging.info("Creating financial ratios transformation pipeline")

            def compute_financial_ratios(df):
                logging.info("Computing financial ratios")
                # Example financial ratio calculations (adjust based on your data)
                df['R1'] = df['Current_Assets'] / df['Current_liabilities']

                # R2: Debt to Equity Ratio = Liabilities / Stockholder Equity
                df['R2'] = df['Liabilities'] / df['Stockholder_Equity']
                # R6: Return on Equity (ROE) = Net Income / Stockholder Equity
                df['R3'] = df['NetIncome'] / df['Stockholder_Equity']

                # R7: Cash Ratio = Cash / Current Liabilities
                df['R4'] = df['Cash'] / df['Current_liabilities']

                # R8: Operating Cash Flow to Total Debt Ratio = Net Cash Operating Activities / Total Liabilities
                df['R5'] = df['NetCash_OperatingActivities'] / df['Liabilities']
                # R13: Long-Term Debt to Total Capitalization = Long-Term Debt / (Long-Term Debt + Stockholder Equity)
                df['R6'] = df['LongTerm_Debt'] / (df['LongTerm_Debt'] + df['Stockholder_Equity'])

                # R16: Financing Cash Flow to Total Debt Ratio = Net Cash Financing Activities / Total Liabilities
                df['R7'] = df['NetCash_FinancingActivities'] / df['Liabilities']
            
                return df

            pipeline = Pipeline(steps=[
                ('financial_ratios', FunctionTransformer(compute_financial_ratios))
            ])
            return pipeline

        except Exception as e:
            raise CustomException(e, sys)

    ### Data Cleaning Pipeline
    def create_cleaning_pipeline(self, missing_value_threshold=55.0):
        try:
            # Pipeline for cleaning and reducing data
            cleaning_pipeline = Pipeline(steps=[
                ('drop_specified_columns', FunctionTransformer(self.drop_specified_columns, kw_args={'columns_to_drop': ['cik', 'ticker', 'accessionNo', 'companyName', 'fy', 'fp', 'form', 'filed']})),
                ('drop_missing_value_columns', FunctionTransformer(self.drop_missing_value_columns, kw_args={'missing_value_threshold': missing_value_threshold}))
            ])
            return cleaning_pipeline
        except Exception as e:
            raise CustomException(e, sys)
        
    def create_missing_value_pipeline(self):
        try:
            # Pipeline for missing value imputation (applied only to minority class)
            pipeline = Pipeline(steps=[
                ('fill_missing_median', FunctionTransformer(lambda X: X.fillna(X.median()))),  # Fill missing values with median
                ('iterative_imputer', IterativeImputer(estimator=LinearRegression(), max_iter=10, random_state=0))  # Apply MICE imputation
            ])
            return pipeline
        except Exception as e:
            raise CustomException(e, sys)
    def create_scaling_pipeline(self):
        try:
            # Pipeline for scaling (applied to both majority and minority classes)
            pipeline = Pipeline(steps=[
                ('scaler', RobustScaler())  # Apply Robust Scaling
            ])
            return pipeline
        except Exception as e:
            raise CustomException(e, sys)

    ### Initiate data transformation
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Step 1: Read train and test data
            logging.info(f"Reading train and test data from {train_path} and {test_path}")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Step 2: Drop duplicate rows
            logging.info(f"Number of duplicate rows removed from train: {train_df.duplicated().sum()}")
            train_df = train_df.drop_duplicates()
            logging.info(f"Number of duplicate rows removed from test: {test_df.duplicated().sum()}")
            test_df = test_df.drop_duplicates()

            # Step 3: Create a cleaning pipeline
            logging.info("Creating a cleaning pipeline")
            cleaning_pipeline = self.create_cleaning_pipeline(missing_value_threshold=55.0)

            # Step 4: Fit the cleaning pipeline on train data and transform both train and test data
            logging.info("Fitting and transforming train data with cleaning pipeline")
            train_df_cleaned = cleaning_pipeline.fit_transform(train_df)
            logging.info("Transforming test data with cleaning pipeline")
            test_df_cleaned = cleaning_pipeline.transform(test_df)

            # Step 5: Save the cleaning pipeline
            logging.info("Saving cleaning pipeline")
            save_object(self.data_transformation_config.cleaning_pipeline_path, cleaning_pipeline)

            # Step 6: Define the target column
            logging.info("Defining target column")
            target_column_name = 'is_bankrupt'

            # Step 7: Split majority and minority classes in train and test data
            logging.info("Splitting majority and minority classes in train data")
            df_majority_train = train_df_cleaned[train_df_cleaned[target_column_name] == 0].dropna()
            df_minority_train = train_df_cleaned[train_df_cleaned[target_column_name] == 1]

            logging.info("Splitting majority and minority classes in test data")
            df_majority_test = test_df_cleaned[test_df_cleaned[target_column_name] == 0].dropna()
            df_minority_test = test_df_cleaned[test_df_cleaned[target_column_name] == 1]
            # Step 8: Handle missing data in the minority class (train)
            logging.info("Handling missing values for minority class in train data")
            missing_value_pipeline = self.create_missing_value_pipeline()

            # Fit the pipeline on the minority class training data
            df_minority_train_imputed = missing_value_pipeline.fit_transform(df_minority_train)

            # Convert the imputed data back to DataFrame with the same columns
            df_minority_train_imputed = pd.DataFrame(df_minority_train_imputed, columns=df_minority_train.columns)

            #Apply the fitted missing value pipeline on the test data without refitting
            df_minority_test_imputed = missing_value_pipeline.transform(df_minority_test)

            # Convert the imputed test data back to DataFrame with the same columns
            df_minority_test_imputed= pd.DataFrame(df_minority_test_imputed, columns=df_minority_test.columns)
            # Combine the majority and imputed minority data
            combined_train_data = pd.concat([df_majority_train, df_minority_train_imputed], ignore_index=True)

            # Optional: Log the shape of the combined data for verification
            logging.info(f"Shape of df_majority_train: {df_majority_train.shape}")
            logging.info(f"Shape of df_minority_train_imputed: {df_minority_train_imputed.shape}")

            logging.info("Checking for null values in majority and minority class data")

            # Log         
            logging.info(f"Checking for NaNs in df_minority_train_imputed data: {df_minority_train_imputed.isna().sum().sum()}")
            logging.info(f"Checking for NaNs in df_minority_train_imputed data: {df_majority_train.isna().sum().sum()}")
            logging.info(f"Checking for NaNs in df_majority_test data: {df_majority_test.isna().sum().sum()}")
            logging.info(f"Checking for NaNs in df_minority_test data: {df_minority_test_imputed.isna().sum().sum()}")

            # Step 9: Combine majority and minority classes before scaling
            logging.info("Combining majority and minority class data")
            combined_train_data = pd.concat([df_majority_train, df_minority_train_imputed])

            # Resetting index for the combined DataFrame
            combined_train_data.reset_index(drop=True, inplace=True)

            # Step 10: Apply scaling to the combined data
            logging.info("Applying scaling to combined training data")
            scaling_pipeline = self.create_scaling_pipeline()

            # Fit the scaling pipeline on the combined training data
            combined_train_scaled = scaling_pipeline.fit_transform(combined_train_data)

            # Convert scaled data back to DataFrame
            combined_train_scaled = pd.DataFrame(combined_train_scaled, columns=combined_train_data.columns)

            # Save the fitted scaling pipeline to a pickle file
            logging.info("Saving scaling pipeline")
            save_object(self.data_transformation_config.scaling_pipeline_path, scaling_pipeline)

            # Step 11: Prepare the scaled training and testing datasets
            logging.info("Preparing test dataset")
            test_combined_data = pd.concat([df_majority_test, df_minority_test_imputed])

            # Resetting index for the combined test DataFrame
            test_combined_data.reset_index(drop=True, inplace=True)

            # Scale the combined test data
            test_combined_scaled = scaling_pipeline.transform(test_combined_data)

            # Convert test scaled data back to DataFrame
            test_combined_scaled = pd.DataFrame(test_combined_scaled, columns=test_combined_data.columns)

            # Debug: Log the shape and columns of the combined DataFrames
            logging.info(f"Shape of combined_train_scaled: {combined_train_scaled.shape}")
            logging.info(f"Shape of test_combined_scaled: {test_combined_scaled.shape}")
            logging.info(f"Checking for NaNs in test_combined_scaled data: {test_combined_scaled.isna().sum().sum()}")
            logging.info(f"Checking for NaNs in combined_train_scaled data: {combined_train_scaled.isna().sum().sum()}")

            # Step 13: Prepare the features and target
            input_feature_train_df = combined_train_scaled.drop(columns=[target_column_name])
            target_feature_train_df = combined_train_scaled[target_column_name]
            input_feature_test_df = test_combined_scaled.drop(columns=[target_column_name])
            target_feature_test_df = test_combined_scaled[target_column_name]

            # Optional: Log the shapes of input and target features
            logging.info(f"Input features shape (train): {input_feature_train_df.shape}")
            logging.info(f"Target feature shape (train): {target_feature_train_df.shape}")
            logging.info(f"Input features shape (test): {input_feature_test_df.shape}")
            logging.info(f"Target feature shape (test): {target_feature_test_df.shape}")

            """ 
            # Step 11: Feature engineering
            logging.info("Starting feature engineering for train and test data")
            feature_engineering_pipeline = self.financial_ratios_pipeline()
            train_df_fe = feature_engineering_pipeline.fit_transform(combined_train_scaled)
            test_df_fe = feature_engineering_pipeline.transform(test_combined_scaled)

            # Step 12: Save the feature engineering pipeline
            logging.info("Saving feature engineering pipeline")
            save_object(self.data_transformation_config.feature_engineering_pipeline_path, feature_engineering_pipeline)

            # Step 13: Prepare the features and target
            input_feature_train_df = train_df_fe.drop(columns=[target_column_name])
            target_feature_train_df = train_df_fe[target_column_name]
            input_feature_test_df = test_df_fe.drop(columns=[target_column_name])
            target_feature_test_df = test_df_fe[target_column_name] """
            
            # Step 14: Apply selected feature selection technique
            logging.info(f"Applying feature selection method: {self.method}")

            if self.method == 'pearson':
                feature_selection_pipeline = self.pearson_correlation_pipeline(input_feature_train_df, target_feature_train_df)
            elif self.method == 'spearman':
                feature_selection_pipeline = self.spearman_correlation_pipeline(input_feature_train_df, target_feature_train_df)
            elif self.method == 'mutual_info':
                feature_selection_pipeline = self.mutual_information_pipeline(target_feature_train_df)
            elif self.method == 'random_forest_rfe':
                feature_selection_pipeline = self.random_forest_rfe_pipeline(target_feature_train_df)
            else:
                logging.error(f"Feature selection method '{self.method}' is not recognized.")
                raise ValueError(f"Feature selection method '{self.method}' is not recognized.")

            # Fit and transform feature selection pipeline
            logging.info("Fitting feature selection pipeline on training data")
            input_feature_train_selected = feature_selection_pipeline.fit_transform(input_feature_train_df, target_feature_train_df)

            logging.info("Transforming test data with the fitted feature selection pipeline")
            logging.info(f"Shape of input_feature_train_df: {input_feature_train_df.shape}")
            logging.info(f"Shape of input_feature_test_df: {input_feature_test_df.shape}")
            logging.info(f"Columns in train: {input_feature_train_df.columns.tolist()}")
            logging.info(f"Columns in test: {input_feature_test_df.columns.tolist()}")
            logging.info(f"Checking for NaNs in test data: {input_feature_test_df.isna().sum().sum()}")
            logging.info(f"Checking for Inf values in test data: {np.isinf(input_feature_test_df).sum().sum()}")

            input_feature_test_selected = feature_selection_pipeline.transform(input_feature_test_df)
            
            # Step 15: Save the feature selection object

            logging.info(f"Saving feature selection object for method: {self.method}")
            save_object(self.data_transformation_config.feature_selec_obj_path, feature_selection_pipeline)

            # Step 16: Perform undersampling using KMeans clustering on successful class
            logging.info("Performing KMeans clustering and undersampling on successful class (majority class)")
            X_train_successful = pd.DataFrame(input_feature_train_selected[target_feature_train_df == 0])
            y_train_successful = pd.DataFrame(target_feature_train_df[target_feature_train_df == 0])

            X_train_ban = pd.DataFrame(input_feature_train_selected[target_feature_train_df == 1])
            y_train_ban = pd.DataFrame(target_feature_train_df[target_feature_train_df == 1])

            n_clusters = 6  # You can adjust this number depending on your data
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            X_train_successful['cluster'] = kmeans.fit_predict(X_train_successful)

            # For each cluster, undersample to get a balanced subset
            n_samples_per_cluster = 33
            undersampled_majority = []

            for cluster in range(n_clusters):
                cluster_data = X_train_successful[X_train_successful['cluster'] == cluster]
                if len(cluster_data) > n_samples_per_cluster:
                    undersampled_cluster = cluster_data.sample(n=n_samples_per_cluster, random_state=42)
                else:
                    undersampled_cluster = cluster_data  # If the cluster has fewer samples, take all of them
                undersampled_majority.append(undersampled_cluster)

            # Concatenate the undersampled clusters
            X_train_successful_undersampled = pd.concat(undersampled_majority).drop(columns=['cluster'])

            # Combine the undersampled majority class with the minority class
            X_train_balanced = pd.concat([X_train_successful_undersampled, X_train_ban])
            y_train_balanced = pd.concat([y_train_successful.loc[X_train_successful_undersampled.index], y_train_ban])

            # Shuffle the data (optional)
            X_train = X_train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
            y_train = y_train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
            logging.info(f"count: {y_train.value_counts()}")

            # Step 17: Apply ADASYN to the training data
            logging.info("Applying ADASYN oversampling")
            adasyn = ADASYN(random_state=42)
            X_train, y_train = adasyn.fit_resample(X_train, y_train)
            logging.info(f"count: {y_train.value_counts()}")

            # Step 18: Return the transformed datasets
            return X_train, y_train, input_feature_test_selected, target_feature_test_df

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
