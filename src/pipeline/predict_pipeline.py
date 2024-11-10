import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging
import numpy as np
import pandas as pd

class PredictPipeline:
    def __init__(self):
        # Load all preprocessing objects and model during initialization
        self.cleaning_pipeline = load_object(file_path=os.path.join("artifacts", "cleaning_pipeline.pkl"))
        self.eng_pipeline = load_object(file_path=os.path.join("artifacts", "feature_engineering.pkl"))
        self.missing_pipeline = load_object(file_path=os.path.join("artifacts", "preprocessor.pkl"))
        self.scaling_pipeline = load_object(file_path=os.path.join("artifacts", "scaling_pipeline.pkl"))
        self.feature_selection_pipeline = load_object(file_path=os.path.join("artifacts", "feature_selection.pkl"))
        self.model = load_object(file_path=os.path.join("artifacts", "classification_model.pkl"))

    def predict(self, features):
        try:
            logging.info(f"Original DataFrame columns: {features.columns.tolist()}")

            # Step 1: Clean the data
            data_cleaned = self.cleaning_pipeline.transform(features)
            if isinstance(data_cleaned, np.ndarray):
                data_cleaned = pd.DataFrame(data_cleaned, columns=features.columns)
            logging.info(f"Data shape after cleaning: {data_cleaned.columns.tolist()}")

            # Step 2: Handle missing data
            data_miss = self.missing_pipeline.transform(data_cleaned)
            if isinstance(data_miss, np.ndarray):
                data_miss = pd.DataFrame(data_miss, columns=data_cleaned.columns)
            logging.info(f"Data shape after missing data handling: {data_miss.columns.tolist()}")

            # Step 3: Scale data
            data_scaled = self.scaling_pipeline.transform(data_miss)
            if isinstance(data_scaled, np.ndarray):
                data_scaled = pd.DataFrame(data_scaled, columns=data_miss.columns)
            logging.info(f"Data shape after scaling: {data_scaled.columns.tolist()}")

            # Step 4: Apply feature engineering
            data_eng = self.eng_pipeline.transform(data_scaled)
            if isinstance(data_eng, np.ndarray):
                data_eng = pd.DataFrame(data_eng, columns=data_scaled.columns)
            logging.info(f"Data shape after feature engineering: {data_eng.shape}")

            # Step 5: Feature selection
            data_selected = self.feature_selection_pipeline.transform(data_eng)
            if isinstance(data_selected, np.ndarray):
                data_selected = pd.DataFrame(data_selected)
            logging.info(f"Data shape after feature selection: {data_selected.shape}")

            # Step 2: Make predictions
            predictions = self.model.predict(data_selected)

            # Step 3: Get prediction probabilities if available
            if hasattr(self.model, "predict_proba"):
                predictions_proba = self.model.predict_proba(data_selected)[:, 1]  # Probability of positive class
            else:
                predictions_proba = [None] * len(predictions)

            return predictions, predictions_proba
        except Exception as e:
            raise CustomException(e, sys)




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
