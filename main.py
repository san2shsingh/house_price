import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib
import sqlite3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Step 1: Load and preprocess input data
def load_data(file_path):
    logger.info("Loading and preprocessing input data from file: %s", file_path)
    data = pd.read_csv(file_path)  # Assuming CSV file, adjust as needed
    # Perform any necessary preprocessing on the data

    # Label encoding for categorical variables
    le = LabelEncoder()
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = le.fit_transform(data[column])

    return data

# Step 2: Transform data for model
def transform_data(data):
    logger.info("Transforming data for the model")
    # Perform transformations as needed
    # For simplicity, let's assume the target variable is named 'price'
    X = data.drop('MEDIAN_HOUSE_VALUE', axis=1)
    y = data['MEDIAN_HOUSE_VALUE']
    return X, y

# Step 3: Train a regression model
def train_model(X_train, y_train):
    logger.info("Training the regression model")
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

# Step 4: Make predictions
def make_predictions(model, X_test):
    logger.info("Making predictions using the trained model")
    predictions = model.predict(X_test)
    return predictions

# Step 5: Evaluate the model
def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating the model")
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    logger.info("Mean Squared Error: %f", mse)

# Step 6: Store model using joblib
def store_model(model, model_filename='model.joblib'):
    logger.info("Storing the trained model using joblib")
    joblib.dump(model, model_filename, compress=3)

# Step 7: Store results in a database
def store_results(predictions, data):
    logger.info("Storing results in the database")
    conn = sqlite3.connect('house_price_predictions.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS house_price_predictions (
            id INTEGER PRIMARY KEY,
            input_data TEXT,
            prediction REAL
        )
    ''')

    for i in range(len(predictions)):
        input_data = str(data.iloc[i].to_dict())
        prediction = predictions[i]
        cursor.execute('''
            INSERT INTO house_price_predictions (input_data, prediction)
            VALUES (?, ?)
        ''', (input_data, prediction))

    conn.commit()
    conn.close()

# Step 8: Execute the pipeline
if __name__ == "__main__":
    # Input data file path
    input_data_file = 'housing.csv'

    try:
        # Step 1: Load and preprocess input data
        input_data = load_data(input_data_file)

        # Step 2: Transform data for model
        X, y = transform_data(input_data)

        # Step 3: Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 100)

        # Step 4: Train a regression model
        trained_model = train_model(X_train, y_train)

        # Step 5: Evaluate the model
        evaluate_model(trained_model, X_test, y_test)

        # Step 6: Store the model using joblib
        store_model(trained_model)

        # Step 7: Make predictions
        predictions = make_predictions(trained_model, X_test)

        # Step 8: Store results in a database
        store_results(predictions, input_data)

        logger.info("Pipeline executed successfully!")

    except Exception as e:
        logger.exception("An error occurred: %s", str(e))
