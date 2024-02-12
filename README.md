# House Price Prediction Web App

This web application allows users to predict house prices based on house property parameters. It uses a RandomForestRegressor model and is built with Flask.

## Requirements

- Python 3.10
- Required Python packages: 
    joblib==1.2.0
    numpy==1.23.4
    pandas==1.5.0
    python-dateutil==2.8.2
    pytz==2022.4
    scikit-learn==1.1.2
    scipy==1.9.2
    six==1.16.0
    sklearn==0.0
    threadpoolctl==3.1.0

Install the required packages using the following command:

pip install -r requirements.txt

# Usage
## Clone the repository:

git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction

1. Place your pre-trained model file (house_price_model.joblib) in the project directory.

2. Run the Flask web application:

python house_api.py

1. Visit http://127.0.0.1:5000/ in your web browser to access the web app. Enter the house property parameters, click the "Predict" button, and view the predictions on the webpage.

## Script Details
main.py: The main script that loads the data, trains a RandomForestRegressor model, makes predictions, evaluates the model, stores the trained model using joblib, and stores the results in a SQLite database.

## Logging
Logging is implemented in the script using the logging module. Log messages provide information about each step of the pipeline. Exceptions are caught and logged in case of errors.

## Database
Results (input data and predictions) are stored in an SQLite database named house_price_predictions.db.
