import numpy as np
import pandas as pd
import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging
import numpy as np


class PredictPipeline:
    def __init__(self):
        # Load all preprocessing objects and model during initialization
        self.missing_pipeline = load_object(file_path=os.path.join("artifacts", "preprocessor.pkl"))
        self.scaling_pipeline = load_object(file_path=os.path.join("artifacts", "scaling_pipeline.pkl"))
        self.feature_selection_pipeline = load_object(file_path=os.path.join("artifacts", "feature_selection.pkl"))
        self.model = load_object(file_path=os.path.join("artifacts", "classification_model.pkl"))

    

    def predict(self, features):
        try:
            logging.info(f"Original DataFrame columns: {features.columns.tolist()}")

            # Step 1: Drop unnecessary columns
            columns_to_drop = [
                'id', 'cik', 'ticker', 'accessionNo', 'companyName', 'fy', 'fp', 'form', 'filed',
                'Current_Other_Assets', 'Nonoperating_Income', 'Intangible_Assets', 'GrossProfit'
            ]
            features = self.drop_specified_columns(features, columns_to_drop)
            logging.info(f"Data shape after dropping specified columns: {features.shape}")

            # Step 2: Handle missing data
            data_miss = self.missing_pipeline.transform(features)
            if isinstance(data_miss, np.ndarray):
                data_miss = pd.DataFrame(data_miss, columns=features.columns)
            logging.info(f"Data shape after missing data handling: {data_miss.shape}")

            # Step 3: Scale data
            data_scaled = self.scaling_pipeline.transform(data_miss)
            if isinstance(data_scaled, np.ndarray):
                data_scaled = pd.DataFrame(data_scaled, columns=data_miss.columns)
            logging.info(f"Data shape after scaling: {data_scaled.shape}")

            # Step 4: Apply feature engineering (ratios calculation)
            data_ratios = self.calculate_ratios(data_scaled)
            data_combined = pd.concat([data_scaled, data_ratios], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            logging.info(f"Data shape after feature engineering: {data_combined.shape}")

            # Step 5: Feature selection
            data_selected = self.feature_selection_pipeline.transform(data_combined)
            if isinstance(data_selected, np.ndarray):
                data_selected = pd.DataFrame(data_selected)
            logging.info(f"Data shape after feature selection: {data_selected.shape}")

            # Step 6: Make predictions
            predictions = self.model.predict(data_selected)

            # Step 7: Get prediction probabilities if available
            if hasattr(self.model, "predict_proba"):
                predictions_proba = self.model.predict_proba(data_selected)[:, 1]  # Probability of positive class
            else:
                predictions_proba = [None] * len(predictions)

            return predictions, predictions_proba
        except Exception as e:
            raise CustomException(e, sys)
    @staticmethod
    def drop_specified_columns(X, columns_to_drop):
        """Drops specified columns from the input DataFrame."""
        return X.drop(columns=[col for col in columns_to_drop if col in X.columns], axis=1)
        
    @staticmethod
    def calculate_ratios(data):
        """Calculate financial ratios directly in the pipeline."""
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

class CustomData:
    def __init__(self, 
                 id: str,
                 companyName: str,
                 ticker: str,
                 accessionNo: str,
                 fy: int,
                 fp: str,
                 form: str,
                 filed: str,
                 assets: float,
                 currentAssets: float,
                 currentLiabilities: float,
                 stockholdersEquity: float,
                 liabilitiesAndStockholdersEquity: float,
                 earningBeforeInterestAndTaxes: float,
                 retainedEarnings: float,
                 revenues: float,
                 workingCapital: float,
                 liabilities: float,
                 netCashOperatingActivities: float,
                 netCashInvestingActivities: float,
                 netCashFinancingActivities: float,
                 cash: float,
                 accountsReceivable: float,
                 inventory: float,
                 currentOtherAssets: float,
                 noncurrentAssets: float,
                 intangibleAssets: float,
                 accountsPayable: float,
                 netIncome: float,
                 grossProfit: float,
                 operatingExpenses: float,
                 nonoperatingIncome: float,
                 interestExpense: float,
                 shortTermDebt: float,
                 longTermDebt: float,
                 noncurrentLiabilities: float,
                 prediction: float = None  # Default to None
                ):
        # Initialize all attributes with provided values
        self.cik = id
        self.companyName = companyName
        self.ticker = ticker
        self.accessionNo = accessionNo
        self.fy = fy
        self.fp = fp
        self.form = form
        self.filed = filed
        self.Assets = assets
        self.Current_Assets = currentAssets
        self.Current_liabilities = currentLiabilities
        self.Stockholder_Equity = stockholdersEquity
        self.Liabilities_And_StockholderEquity = liabilitiesAndStockholdersEquity
        self.Earning_Before_Interest_And_Taxes = earningBeforeInterestAndTaxes
        self.Retained_Earnings = retainedEarnings
        self.Revenues = revenues
        self.Working_capital = workingCapital
        self.Liabilities = liabilities
        self.NetCash_OperatingActivities = netCashOperatingActivities
        self.NetCash_InvestingActivities = netCashInvestingActivities
        self.NetCash_FinancingActivities = netCashFinancingActivities
        self.Cash = cash
        self.AccountsReceivable = accountsReceivable
        self.Inventory = inventory
        self.Current_Other_Assets = currentOtherAssets
        self.Noncurrent_Assets = noncurrentAssets
        self.Intangible_Assets = intangibleAssets
        self.AccountsPayable = accountsPayable
        self.NetIncome = netIncome
        self.GrossProfit = grossProfit
        self.Operating_Expenses = operatingExpenses
        self.Nonoperating_Income = nonoperatingIncome
        self.InterestExpense = interestExpense
        self.ShortTerm_Debt = shortTermDebt
        self.LongTerm_Debt = longTermDebt
        self.Noncurrent_Liabilities = noncurrentLiabilities
        self.prediction = prediction  # Optional; used only for storage after inference

    def get_data_as_data_frame(self):
        try:
            # Create a dictionary with all attributes as key-value pairs, excluding `prediction`
            custom_data_input_dict = {
                "cik": [self.cik],
                "companyName": [self.companyName],
                "ticker": [self.ticker],
                "accessionNo": [self.accessionNo],
                "fy": [self.fy],
                "fp": [self.fp],
                "form": [self.form],
                "filed": [self.filed],
                "Assets": [self.Assets],
                "Current_Assets": [self.Current_Assets],
                "Current_liabilities": [self.Current_liabilities],
                "Stockholder_Equity": [self.Stockholder_Equity],
                "Liabilities_And_StockholderEquity": [self.Liabilities_And_StockholderEquity],
                "Earning_Before_Interest_And_Taxes": [self.Earning_Before_Interest_And_Taxes],
                "Retained_Earnings": [self.Retained_Earnings],
                "Revenues": [self.Revenues],
                "Working_capital": [self.Working_capital],
                "Liabilities": [self.Liabilities],
                "NetCash_OperatingActivities": [self.NetCash_OperatingActivities],
                "NetCash_InvestingActivities": [self.NetCash_InvestingActivities],
                "NetCash_FinancingActivities": [self.NetCash_FinancingActivities],
                "Cash": [self.Cash],
                "AccountsReceivable": [self.AccountsReceivable],
                "Inventory": [self.Inventory],
                "Current_Other_Assets": [self.Current_Other_Assets],
                "Noncurrent_Assets": [self.Noncurrent_Assets],
                "Intangible_Assets": [self.Intangible_Assets],
                "AccountsPayable": [self.AccountsPayable],
                "NetIncome": [self.NetIncome],
                "GrossProfit": [self.GrossProfit],
                "Operating_Expenses": [self.Operating_Expenses],
                "Nonoperating_Income": [self.Nonoperating_Income],
                "InterestExpense": [self.InterestExpense],
                "ShortTerm_Debt": [self.ShortTerm_Debt],
                "LongTerm_Debt": [self.LongTerm_Debt],
                "Noncurrent_Liabilities": [self.Noncurrent_Liabilities],
            }

            # Return a DataFrame created from the dictionary
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
