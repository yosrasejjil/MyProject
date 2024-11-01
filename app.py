from flask import Flask, request, jsonify
import pandas as pd
from src.pipeline import PredictPipeline  # Import your pipeline class here
app = Flask(__name__)

# Initialize your prediction pipeline
predict_pipeline = PredictPipeline()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Convert the JSON into a Pandas DataFrame
        input_data = pd.DataFrame([data])
        
        # Make predictions using your pipeline
        predictions = predict_pipeline.predict(input_data)
        
        # Send back the predictions as a JSON response
        return jsonify({
            'predictions': predictions.tolist()
        })

    except Exception as e:
        # Handle any exceptions
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Flask will run on port 5000
""" 
import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging
class PredictPipeline:
    def __init__(self):
        # Load all necessary pipelines
        self.cleaning_pipeline = load_object(os.path.join("artifacts", "cleaning_pipeline.pkl"))
        self.scaling_pipeline = load_object(os.path.join("artifacts", "scaling_pipeline.pkl"))
        self.feature_selection_pipeline = load_object(os.path.join("artifacts", "feature_selection_pipeline.pkl"))
        self.model = load_object(os.path.join("artifacts", "model.pkl"))  # Update the model name if needed

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Log the initial input features
            logging.info(f"Input features: {features}")

            # Paths to the preprocessing steps and classification model pickle files
            cleaning_pipeline_path = os.path.join("artifacts", "cleaning_pipeline.pkl")
            scaling_pipeline_path = os.path.join('artifacts', "scaling_pipeline.pkl")
            feature_selection_pipeline_path = os.path.join('artifacts', "feature_selection.pkl")
            model_path = os.path.join("artifacts", "classification_model.pkl")

            logging.info("Before loading preprocessing pipelines and model.")

            # Load all preprocessing objects and model
            cleaning_pipeline = load_object(file_path=cleaning_pipeline_path)
            scaling_pipeline = load_object(file_path=scaling_pipeline_path)
            feature_selection_pipeline = load_object(file_path=feature_selection_pipeline_path)
            model = load_object(file_path=model_path)

            logging.info("After loading preprocessing pipelines and model.")

            

            # Convert the formatted data to a DataFrame
            formatted_df = pd.DataFrame(features, index=[0])
            logging.info(f"Formatted DataFrame created with shape: {formatted_df.shape}")

            # Step 1: Clean the data
            logging.info("Cleaning the data...")
            data_cleaned = cleaning_pipeline.transform(formatted_df)
            logging.info(f"Data cleaned. Shape after cleaning: {data_cleaned.shape}")

            # Step 2: Scale the data
            logging.info("Scaling the data...")
            data_scaled = scaling_pipeline.transform(data_cleaned)
            logging.info(f"Data scaled. Shape after scaling: {data_scaled.shape}")

            # Step 3: Apply feature selection
            logging.info("Selecting features...")
            data_selected = feature_selection_pipeline.transform(data_scaled)
            logging.info(f"Feature selection completed. Shape after selection: {data_selected.shape}")

            # Check if any data is selected
            if data_selected.empty:
                logging.error("Feature selection resulted in an empty DataFrame. Please check your feature selection pipeline.")
                raise ValueError("Feature selection resulted in an empty DataFrame.")

            # Step 4: Make predictions with the classification model
            logging.info("Making predictions with the model...")
            predictions = model.predict(data_selected)
            logging.info(f"Predictions made successfully: {predictions}")

            # Store the prediction in the input features DataFrame
            features['prediction'] = predictions[0]

            return predictions
        
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, cik: str, companyName: str, ticker: str, accessionNo: str,
                 fy: int, fp: str, form: str, filed: str, Assets: float,
                 Current_Assets: float, Current_liabilities: float,
                 Stockholder_Equity: float, Liabilities_And_StockholderEquity: float,
                 Earning_Before_Interest_And_Taxes: float, Retained_Earnings: float,
                 Revenues: float, Working_capital: float, Liabilities: float,
                 NetCash_OperatingActivities: float, NetCash_InvestingActivities: float,
                 NetCash_FinancingActivities: float, Cash: float,
                 AccountsReceivable: float, Inventory: float, Current_Other_Assets: float,
                 Noncurrent_Assets: float, Intangible_Assets: float, AccountsPayable: float,
                 NetIncome: float, GrossProfit: float, Operating_Expenses: float,
                 Nonoperating_Income: float, InterestExpense: float,
                 ShortTerm_Debt: float, LongTerm_Debt: float, Noncurrent_Liabilities: float):
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

    def get_data_as_data_frame(self):
        # Convert to a dictionary and then to a DataFrame
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
            "Noncurrent_Liabilities": [self.Noncurrent_Liabilities]
        }

        return pd.DataFrame(custom_data_input_dict)
from flask import Flask, request, jsonify
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging  # Assuming you have a logging module set up

app = Flask(__name__)

# Initialize your prediction pipeline
predict_pipeline = PredictPipeline()

# Route for the home page (you can modify this as needed)
@app.route('/')
def index():
    return jsonify({"message": "Welcome to the prediction API!"})

# Route for predicting data points
@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    try:
        # Collect data from the request
        data = CustomData(
            cik=request.json.get('cik'),
            companyName=request.json.get('companyName'),
            ticker=request.json.get('ticker'),
            accessionNo=request.json.get('accessionNo'),
            fy=int(request.json.get('fy')),
            fp=request.json.get('fp'),
            form=request.json.get('form'),
            filed=request.json.get('filed'),
            Assets=float(request.json.get('Assets')),
            Current_Assets=float(request.json.get('Current_Assets')),
            Current_liabilities=float(request.json.get('Current_liabilities')),
            Stockholder_Equity=float(request.json.get('Stockholder_Equity')),
            Liabilities_And_StockholderEquity=float(request.json.get('Liabilities_And_StockholderEquity')),
            Earning_Before_Interest_And_Taxes=float(request.json.get('Earning_Before_Interest_And_Taxes')),
            Retained_Earnings=float(request.json.get('Retained_Earnings')),
            Revenues=float(request.json.get('Revenues')),
            Working_capital=float(request.json.get('Working_capital')),
            Liabilities=float(request.json.get('Liabilities')),
            NetCash_OperatingActivities=float(request.json.get('NetCash_OperatingActivities')),
            NetCash_InvestingActivities=float(request.json.get('NetCash_InvestingActivities')),
            NetCash_FinancingActivities=float(request.json.get('NetCash_FinancingActivities')),
            Cash=float(request.json.get('Cash')),
            AccountsReceivable=float(request.json.get('AccountsReceivable')),
            Inventory=float(request.json.get('Inventory')),
            Current_Other_Assets=float(request.json.get('Current_Other_Assets')),
            Noncurrent_Assets=float(request.json.get('Noncurrent_Assets')),
            Intangible_Assets=float(request.json.get('Intangible_Assets')),
            AccountsPayable=float(request.json.get('AccountsPayable')),
            NetIncome=float(request.json.get('NetIncome')),
            GrossProfit=float(request.json.get('GrossProfit')),
            Operating_Expenses=float(request.json.get('Operating_Expenses')),
            Nonoperating_Income=float(request.json.get('Nonoperating_Income')),
            InterestExpense=float(request.json.get('InterestExpense')),
            ShortTerm_Debt=float(request.json.get('ShortTerm_Debt')),
            LongTerm_Debt=float(request.json.get('LongTerm_Debt')),
            Noncurrent_Liabilities=float(request.json.get('Noncurrent_Liabilities'))
        )

        # Convert to DataFrame
        pred_df = data.get_data_as_data_frame()
        logging.info(f"Data for prediction: {pred_df}")

        # Make predictions
        results = predict_pipeline.predict(pred_df)
        logging.info(f"Prediction results: {results}")

        # Return results as JSON
        return jsonify({"predictions": results.tolist()})

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)  # Enable debug for development """