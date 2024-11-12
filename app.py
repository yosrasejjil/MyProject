
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from src.pipeline import PredictPipeline
from src.pipeline.predict_pipeline import CustomData  # Import your pipeline class here


app = Flask(__name__)

# Initialize your prediction pipeline
predict_pipeline = PredictPipeline()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        data.pop('prediction', None)
        custom_data_instance = CustomData(**data)
        input_data = custom_data_instance.get_data_as_data_frame()

        # Call predict method
        predictions, predictions_proba = predict_pipeline.predict(input_data)

        # Convert predictions and probabilities to JSON-compatible formats
        predictions = predictions.tolist() if isinstance(predictions, (np.ndarray, pd.Series)) else predictions
        predictions_proba = [float(prob) if prob is not None else None for prob in predictions_proba]
        print(f"Prediction: {predictions[0]}")
        print(f"Prediction: {predictions_proba[0]}")

        # Send response
        return jsonify({
            'prediction': float(predictions[0]),
            'prediction_proba': predictions_proba[0] if predictions_proba else None
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Flask will run on port 5000

""" 

app = Flask(__name__)

# Initialize your prediction pipeline
predict_pipeline = PredictPipeline()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Remove 'prediction' from the data dictionary if it exists
        data.pop('prediction', None)

        # Create an instance of CustomData with the filtered data
        custom_data_instance = CustomData(**data)
        input_data = custom_data_instance.get_data_as_data_frame()

        # Make predictions using your pipeline
        predictions = predict_pipeline.predict(input_data)

        # Convert predictions to list if they are in a NumPy format
        predictions = predictions.tolist() if isinstance(predictions, (np.ndarray, pd.Series)) else predictions
        print(f"Prediction: {predictions[0]}")

        # Send back the first prediction as a JSON response
        return jsonify({
            'prediction': float(predictions[0])  # Ensure it’s a float for compatibility
        })

    except Exception as e:
        # Handle any exceptions
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    """