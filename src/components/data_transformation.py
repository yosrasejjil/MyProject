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

    def financial_ratios_pipeline(self):
        try:
            logging.info("Creating financial ratios transformation pipeline")
            def compute_financial_ratios(df):
                # df=pd.DataFrame()
                    # R1: Current Ratio = Current Assets / Current Liabilities
                    df['R1'] = df['Current_Assets'] / df['Current_liabilities']

                    # R2: Debt to Equity Ratio = Liabilities / Stockholder Equity
                    df['R2'] = df['Liabilities'] / df['Stockholder_Equity']

                    # R3: Working Capital Ratio = Working Capital / Total Assets
                    df['R3'] = df['Working_capital'] / df['Assets']

                    # R4: Net Income Margin = Net Income / Revenues
                    df['R4'] = df['NetIncome'] / df['Revenues']

                    # R5: Return on Assets (ROA) = Net Income / Total Assets
                    df['R5'] = df['NetIncome'] / df['Assets']

                    # R6: Return on Equity (ROE) = Net Income / Stockholder Equity
                    df['R6'] = df['NetIncome'] / df['Stockholder_Equity']

                    # R7: Cash Ratio = Cash / Current Liabilities
                    df['R7'] = df['Cash'] / df['Current_liabilities']

                    # R8: Operating Cash Flow to Total Debt Ratio = Net Cash Operating Activities / Total Liabilities
                    df['R8'] = df['NetCash_OperatingActivities'] / df['Liabilities']

                    # R9: Interest Coverage Ratio = Earnings Before Interest and Taxes (EBIT) / Interest Expense
                    df['R9'] = df['Earning_Before_Interest_And_Taxes'] / df['InterestExpense']

                    # R10: Debt to Assets Ratio = Liabilities / Total Assets
                    df['R10'] = df['Liabilities'] / df['Assets']

                    # R11: Net Working Capital to Revenues = Working Capital / Revenues
                    df['R11'] = df['Working_capital'] / df['Revenues']

                    # R12: Retained Earnings to Assets Ratio = Retained Earnings / Total Assets
                    df['R12'] = df['Retained_Earnings'] / df['Assets']

                    # R13: Long-Term Debt to Total Capitalization = Long-Term Debt / (Long-Term Debt + Stockholder Equity)
                    df['R13'] = df['LongTerm_Debt'] / (df['LongTerm_Debt'] + df['Stockholder_Equity'])

                    # R14: Cash Flow to Sales Ratio = Net Cash Operating Activities / Revenues
                    df['R14'] = df['NetCash_OperatingActivities'] / df['Revenues']

                    # R15: Investing Cash Flow to Assets Ratio = Net Cash Investing Activities / Total Assets
                    df['R15'] = df['NetCash_InvestingActivities'] / df['Assets']

                    # R16: Financing Cash Flow to Total Debt Ratio = Net Cash Financing Activities / Total Liabilities
                    df['R16'] = df['NetCash_FinancingActivities'] / df['Liabilities']

                    return df

                # # AV3: Total Cash
                # df['AV3'] = df['Cash']
                
                # # AV4: Operating Cash
                # df['AV4'] = df['NetCash_OperatingActivities']
                
                # # AV5: Investing Cash
                # df['AV5'] = df['NetCash_InvestingActivities']
                
                # # AV6: Financing Cash
                # df['AV6'] = df['NetCash_FinancingActivities']
            
            pipeline = Pipeline(steps=[
                ('financial_ratios', FunctionTransformer(compute_financial_ratios))
            ])
            return pipeline

        except Exception as e:
            raise CustomException(e, sys) 

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
                        'id', 'cik', 'ticker', 'accessionNo', 'companyName', 'fy', 'fp', 'form', 'filed', 
                        'Noncurrent_Assets', 'Noncurrent_Liabilities', 'Current_Other_Assets', 'ShortTerm_Debt', 
                        'Nonoperating_Income', 'Intangible_Assets', 'GrossProfit', 'Inventory', 'Operating_Expenses'
                    ]})
                ),
                # Uncomment and adjust the following line if you need to drop columns based on missing values
                # ('drop_missing_value_columns', FunctionTransformer(self.drop_missing_value_columns, kw_args={'missing_value_threshold': missing_value_threshold}))
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
            logging.info(f"Data shape after missing data handling: {train_df_cleaned.columns.tolist()}")
            logging.info(f"Data : {train_df_cleaned.columns.tolist()}")

            # Step 5: Save the cleaning pipeline
            logging.info("Saving cleaning pipeline")
            save_object(self.data_transformation_config.cleaning_pipeline_path, cleaning_pipeline)
            # Step 6: Define the target column
            logging.info("Defining target column")
            target_column_name = 'is_bankrupt'

            # Step 7: Split majority and minority classes in train and test data
            logging.info("Splitting majority and minority classes in train data")
            df_majority_train = train_df_cleaned[train_df_cleaned[target_column_name] == 0].dropna().reset_index(drop=True)
            df_minority_train = train_df_cleaned[train_df_cleaned[target_column_name] == 1].reset_index(drop=True)

            logging.info("Splitting majority and minority classes in test data")
            df_majority_test = test_df_cleaned[test_df_cleaned[target_column_name] == 0].dropna().reset_index(drop=True)
            df_minority_test = test_df_cleaned[test_df_cleaned[target_column_name] == 1].reset_index(drop=True)

            # Separate the target column from minority and majority data
            y_train_majority = df_majority_train[target_column_name]
            y_train_minority = df_minority_train[target_column_name]

            y_test_majority = df_majority_test[target_column_name]
            y_test_minority = df_minority_test[target_column_name]

            # Drop the target column before processing
            df_majority_train = df_majority_train.drop(columns=[target_column_name])
            df_minority_train = df_minority_train.drop(columns=[target_column_name])

            df_majority_test = df_majority_test.drop(columns=[target_column_name])
            df_minority_test = df_minority_test.drop(columns=[target_column_name])

            # Step 8: Handle missing data in the minority class (train)
            logging.info("Handling missing values for minority class in train data")
            missing_value_pipeline = self.create_missing_value_pipeline()

            # Fit the pipeline on the minority class training data
            df_minority_train_imputed = missing_value_pipeline.fit_transform(df_minority_train)
            df_minority_train_imputed = pd.DataFrame(df_minority_train_imputed, columns=df_minority_train.columns)

            # Transform the minority class test data
            df_minority_test_imputed = missing_value_pipeline.transform(df_minority_test)
            df_minority_test_imputed = pd.DataFrame(df_minority_test_imputed, columns=df_minority_test.columns)
            save_object(self.data_transformation_config.missing_path, missing_value_pipeline)

            # Step 9: Combine majority and imputed minority data
            combined_train_data = pd.concat([df_majority_train, df_minority_train_imputed], ignore_index=True)
            combined_test_data = pd.concat([df_majority_test, df_minority_test_imputed], ignore_index=True)

            # Combine target values for train and test data
            y_train = pd.concat([y_train_majority, y_train_minority], ignore_index=True)
            y_test = pd.concat([y_test_majority, y_test_minority], ignore_index=True)

            # Step 10: Drop NaNs in features and targets to maintain consistency
            combined_train_data.dropna(inplace=True)
            y_train = y_train[combined_train_data.index]

            combined_test_data.dropna(inplace=True)
            y_test = y_test[combined_test_data.index]

            # Proceed with scaling and feature engineering as usual
            logging.info("Applying scaling to combined training data")
            scaling_pipeline = self.create_scaling_pipeline()
            combined_train_scaled = scaling_pipeline.fit_transform(combined_train_data)
            combined_train_scaled = pd.DataFrame(combined_train_scaled, columns=combined_train_data.columns)

            # Scale the combined test data
            test_combined_scaled = scaling_pipeline.transform(combined_test_data)
            test_combined_scaled = pd.DataFrame(test_combined_scaled, columns=combined_test_data.columns)

            # Save the scaling pipeline
            logging.info("Saving scaling pipeline")
            save_object(self.data_transformation_config.scaling_pipeline_path, scaling_pipeline)

            # Feature engineering without the target column
            logging.info("Starting feature engineering for train and test data")
            feature_engineering_pipeline = self.financial_ratios_pipeline()            
            train_df_fe = pd.DataFrame(feature_engineering_pipeline.fit_transform(combined_train_scaled), columns=combined_train_scaled.columns)
            test_df_fe = pd.DataFrame(feature_engineering_pipeline.transform(test_combined_scaled), columns=test_combined_scaled.columns)
            save_object(self.data_transformation_config.feature_engineering_pipeline_path, feature_engineering_pipeline)

            logging.info(f" features shape (train): {train_df_fe.shape}")
            logging.info(f" feature shape (test): {test_df_fe.shape}")
            
            logging.info(f"Checking for NaNs in test_combined_scaled data: {train_df_fe.isna().sum().sum()}")
            logging.info(f"Checking for NaNs in combined_train_scaled data: {test_df_fe.isna().sum().sum()}")
            
            #12a: Drop NaN and infinite values from train and test sets
            logging.info("Dropping NaN and infinite values from the engineered features")

            # Replace infinite values with NaN, then drop NaN values in train and test feature sets
            train_df_fe.replace([np.inf, -np.inf], np.nan, inplace=True)
            train_df_fe.dropna(inplace=True)
            test_df_fe.replace([np.inf, -np.inf], np.nan, inplace=True)
            test_df_fe.dropna(inplace=True)

           
            # Align target feature DataFrames with the cleaned input features
            target_feature_train_df = y_train.loc[train_df_fe.index].reset_index(drop=True)
            target_feature_test_df = y_test.loc[test_df_fe.index].reset_index(drop=True)

            # Log the shapes to ensure alignment
            logging.info(f"Input features shape (train) after NaN removal: {train_df_fe.shape}")
            logging.info(f"Target features shape (train) after NaN removal: {target_feature_train_df.shape}")
            logging.info(f"Input features shape (test) after NaN removal: {test_df_fe.shape}")
            logging.info(f"Target features shape (test) after NaN removal: {target_feature_test_df.shape}")

            # Set up the final input features for modeling
            input_feature_train_df = train_df_fe
            input_feature_test_df = test_df_fe

            # Now proceed to feature selection as usual
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

            # Fit and transform feature selection pipeline on the aligned training data
            logging.info("Fitting feature selection pipeline on training data")
            input_feature_train_selected = feature_selection_pipeline.fit_transform(input_feature_train_df, target_feature_train_df)

            logging.info("Transforming test data with the fitted feature selection pipeline")
            input_feature_test_selected = feature_selection_pipeline.transform(input_feature_test_df)

            # Save the feature selection object
            logging.info(f"Saving feature selection object for method: {self.method}")
            save_object(self.data_transformation_config.feature_selec_obj_path, feature_selection_pipeline)


            # Step 16: Perform undersampling using KMeans clustering on successful class
            # Define constants
            n_clusters = 6  # Number of clusters for KMeans
            n_samples_per_cluster = 50  # Desired number of samples per cluster
            # Convertir les ndarrays en DataFrame avant la concaténation
            input_feature_train_selected = pd.DataFrame(input_feature_train_selected)
            input_feature_test_selected = pd.DataFrame(input_feature_test_selected)

            # Vérifier les dimensions pour s'assurer qu'elles sont cohérentes
            logging.info(f"Shape of input_feature_train_selected: {input_feature_train_selected.shape}")
            logging.info(f"Shape of input_feature_test_selected: {input_feature_test_selected.shape}")

            # Combine the training and test datasets
            X_combined = pd.concat([input_feature_train_selected, input_feature_test_selected]).reset_index(drop=True)
            y_combined = pd.concat([target_feature_train_df, target_feature_test_df]).reset_index(drop=True)

            # Separate the majority and minority classes
            X_majority = X_combined[y_combined == 0].copy()
            y_majority = y_combined[y_combined == 0].copy()
            X_minority = X_combined[y_combined == 1].copy()
            y_minority = y_combined[y_combined == 1].copy()

            # Perform KMeans clustering on the majority class
            kmeans = KMeans(n_clusters=n_clusters,n_init='auto', random_state=42)
            X_majority['cluster'] = kmeans.fit_predict(X_majority)

            # Undersample from each cluster
            undersampled_majority = []
            total_samples = 0  # Track total number of undersampled majority samples
            for cluster in range(n_clusters):
                cluster_data = X_majority[X_majority['cluster'] == cluster]
                if len(cluster_data) > n_samples_per_cluster:
                    undersampled_cluster = cluster_data.sample(n=n_samples_per_cluster, random_state=42)
                    logging.info(f"Cluster {cluster} has {len(cluster_data)} samples. Undersampling to {n_samples_per_cluster}.")
                else:
                    undersampled_cluster = cluster_data  # Take all samples if fewer than n_samples_per_cluster
                    logging.warning(f"Cluster {cluster} has only {len(cluster_data)} samples, not enough to undersample to {n_samples_per_cluster}.")
                total_samples += len(undersampled_cluster)
                undersampled_majority.append(undersampled_cluster)

            # Combine undersampled clusters
            X_majority_undersampled = pd.concat(undersampled_majority).drop(columns=['cluster']).reset_index(drop=True)
            y_majority_undersampled = y_majority.loc[X_majority_undersampled.index].reset_index(drop=True)

            logging.info(f"Total undersampled majority samples after combining clusters: {total_samples}")
            logging.info(f"Shape of X_majority_undersampled: {X_majority_undersampled.shape}")
            logging.info(f"Shape of y_majority_undersampled: {y_majority_undersampled.shape}")
            # Combine with minority class
            X_balanced = pd.concat([X_majority_undersampled, X_minority]).reset_index(drop=True)
            y_balanced = pd.concat([y_majority_undersampled, y_minority]).reset_index(drop=True)

            # Log the shapes before shuffling
            logging.info(f"Shape of X_balanced before shuffling: {X_balanced.shape}")
            logging.info(f"Shape of y_balanced before shuffling: {y_balanced.shape}")

            # Shuffle the dataset to ensure a randomized distribution
            X_balanced = X_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
            y_balanced = y_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

            # Determine the split ratio
            train_size = len(input_feature_train_selected) / (len(input_feature_train_selected) + len(input_feature_test_selected))

            # Split the balanced dataset back into training and test sets
            X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(
                X_balanced, y_balanced, train_size=train_size, random_state=42, stratify=y_balanced
            )

            # Log the final class distributions for training and test sets
            logging.info(f"Final Training set count (after balancing and splitting): {y_train_balanced.value_counts()}")
            logging.info(f"Final Test set count (after balancing and splitting): {y_test_balanced.value_counts()}")

            # If you are applying ADASYN or another resampling method on the training set, do it here
            # For example, applying ADASYN to handle any remaining imbalance in the training set
            adasyn = ADASYN(random_state=42)
            X_train, y_train = adasyn.fit_resample(X_train_balanced, y_train_balanced)

            # Check the new class distribution after applying ADASYN
            logging.info(f"Class distribution after ADASYN: {y_train.value_counts()}")
            
            # Step 18: Return the transformed datasets
            return X_train, y_train,X_test_balanced,y_test_balanced

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
