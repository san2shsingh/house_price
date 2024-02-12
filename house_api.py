from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

# Load the pre-trained model
trained_model = joblib.load('model.joblib')

# Label encoder for categorical variables
le = LabelEncoder()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        input_data = {
            'longitude': request.form['longitude'],
            'latitude': request.form['latitude'],
            'housing_median_age': request.form['housing_median_age'],
            'total_rooms': request.form['total_rooms'],
            'total_bedrooms': request.form['total_bedrooms'],
            'population': request.form['population'],
            'households': request.form['households'],
            'median_income': request.form['median_income'],
            'ocean_proximity': request.form['ocean_proximity'],
            'agency': request.form['agency'],   
        }

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Label encoding for categorical variables
        for column in input_df.select_dtypes(include=['object']).columns:
            input_df[column] = le.fit_transform(input_df[column])

        # Make predictions
        prediction = trained_model.predict(input_df)[0]

        return render_template('index.html', prediction=f'Predicted House Price: {prediction}')

    except Exception as e:
        return render_template('index.html', error=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
