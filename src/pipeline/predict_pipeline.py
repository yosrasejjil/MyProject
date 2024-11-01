import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info(f"Input DataFrame columns: {features.columns.tolist()}")

            # Paths to the preprocessing steps and classification model pickle files
            cleaning_pipeline_path = os.path.join("artifacts", "cleaning_pipeline.pkl")
            scaling_pipeline_path = os.path.join('artifacts', "scaling_pipeline.pkl")
            feature_selection_pipeline_path = os.path.join('artifacts', "feature_selection.pkl")
            model_path = os.path.join("artifacts", "classification_model.pkl")

            print("Before Loading")

            # Load all preprocessing objects and model
            cleaning_pipeline = load_object(file_path=cleaning_pipeline_path)
            scaling_pipeline = load_object(file_path=scaling_pipeline_path)
            feature_selection_pipeline = load_object(file_path=feature_selection_pipeline_path)
            model = load_object(file_path=model_path)

            print("After Loading")

            # Step 1: Standardize column names
            # Convert all column names to lowercase with underscores for consistency
            features.columns = features.columns.str.lower().str.replace(' ', '_')

            # Step 2: Clean the data
            data_cleaned = cleaning_pipeline.transform(features)

            # Step 3: Scale the data
            data_scaled = scaling_pipeline.transform(data_cleaned)

            # Step 4: Apply feature selection
            data_selected = feature_selection_pipeline.transform(data_scaled)

            # Step 5: Make predictions with the classification model
            predictions = model.predict(data_selected)

            return predictions

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, 
                 cik: str,
                 companyName: str,
                 ticker: str,
                 accessionNo: str,
                 fy: int,
                 fp: str,
                 form: str,
                 filed: str,
                 Assets: float,
                 Current_Assets: float,
                 Current_liabilities: float,
                 Stockholder_Equity: float,
                 Liabilities_And_StockholderEquity: float,
                 Earning_Before_Interest_And_Taxes: float,
                 Retained_Earnings: float,
                 Revenues: float,
                 Working_capital: float,
                 Liabilities: float,
                 NetCash_OperatingActivities: float,
                 NetCash_InvestingActivities: float,
                 NetCash_FinancingActivities: float,
                 Cash: float,
                 AccountsReceivable: float,
                 Inventory: float,
                 Current_Other_Assets: float,
                 Noncurrent_Assets: float,
                 Intangible_Assets: float,
                 AccountsPayable: float,
                 NetIncome: float,
                 GrossProfit: float,
                 Operating_Expenses: float,
                 Nonoperating_Income: float,
                 InterestExpense: float,
                 ShortTerm_Debt: float,
                 LongTerm_Debt: float,
                 Noncurrent_Liabilities: float,
                 is_bankrupt: int  # Assuming this is the target (bankruptcy flag)
                ):
        # Initialize all attributes with provided values
        self.cik = cik
        self.companyName = companyName
        self.ticker = ticker
        self.accessionNo = accessionNo
        self.fy = fy
        self.fp = fp
        self.form = form
        self.filed = filed
        self.Assets = Assets
        self.Current_Assets = Current_Assets
        self.Current_liabilities = Current_liabilities
        self.Stockholder_Equity = Stockholder_Equity
        self.Liabilities_And_StockholderEquity = Liabilities_And_StockholderEquity
        self.Earning_Before_Interest_And_Taxes = Earning_Before_Interest_And_Taxes
        self.Retained_Earnings = Retained_Earnings
        self.Revenues = Revenues
        self.Working_capital = Working_capital
        self.Liabilities = Liabilities
        self.NetCash_OperatingActivities = NetCash_OperatingActivities
        self.NetCash_InvestingActivities = NetCash_InvestingActivities
        self.NetCash_FinancingActivities = NetCash_FinancingActivities
        self.Cash = Cash
        self.AccountsReceivable = AccountsReceivable
        self.Inventory = Inventory
        self.Current_Other_Assets = Current_Other_Assets
        self.Noncurrent_Assets = Noncurrent_Assets
        self.Intangible_Assets = Intangible_Assets
        self.AccountsPayable = AccountsPayable
        self.NetIncome = NetIncome
        self.GrossProfit = GrossProfit
        self.Operating_Expenses = Operating_Expenses
        self.Nonoperating_Income = Nonoperating_Income
        self.InterestExpense = InterestExpense
        self.ShortTerm_Debt = ShortTerm_Debt
        self.LongTerm_Debt = LongTerm_Debt
        self.Noncurrent_Liabilities = Noncurrent_Liabilities
        self.is_bankrupt = is_bankrupt
        self.prediction = None  # Default to None until prediction is made

    def get_data_as_data_frame(self):
        try:
            # Create a dictionary with all attributes as key-value pairs
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
                "is_bankrupt": [self.is_bankrupt]
            }

            # Return a DataFrame created from the dictionary
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
