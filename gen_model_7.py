import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request
import numpy as np


app = Flask(__name__)

# Initialize encoders
sector_encoder = LabelEncoder()
risk_type_encoder = LabelEncoder()
stock_type_encoder = LabelEncoder()

def generate_random_data():
    """Generates random data for training."""
    data = [
        [28.52, 'Oil & Gas', 'Moderate', 14.37, 'RELIANCE'],
        [29.02, 'Oil & Gas', 'Low', 34.37, 'MobilOil'],
        [29.02, 'Oil & Gas', 'High', 34.37, 'Aramco'],
        [29.84, 'IT Services', 'Low', 21.13, 'TCS'],
        [30.67, 'Banking', 'Low', 16.43, 'HDFCBANK'],
        [29.11, 'IT Services', 'Moderate', 18.69, 'INFY'],
        [24.85, 'Banking', 'Moderate', 11.76, 'ICICIBANK'],
        [11.73, 'Banking', 'High', 8.52, 'SBIN'],
        [31.92, 'Banking', 'Moderate', 14.48, 'KOTAKBANK'],
        [67.92, 'FMCG', 'Low', 4.63, 'HINDUNILVR'],
        [21.27, 'Infrastructure', 'Moderate', 9.32, 'LT'],
        [25.56, 'Banking', 'Moderate', 12.52, 'AXISBANK'],
        [35.41, 'Healthcare', 'Low', 8.94, 'DRREDDY'],
        [41.18, 'Healthcare', 'Low', 6.71, 'SUNPHARMA'],
        [37.62, 'Healthcare', 'Moderate', 5.43, 'CIPLA'],
        [45.28, 'Healthcare', 'Moderate', 7.26, 'AUROPHARMA'],
        [39.77, 'Healthcare', 'High', 4.95, 'DIVISLAB'],
        [32.94, 'Finance', 'Low', 11.68, 'BAJFINANCE'],
        [30.87, 'Finance', 'Moderate', 9.82, 'CHOLAFIN'],
        [36.75, 'Finance', 'Moderate', 7.55, 'LICHSGFIN'],
        [35.06, 'Finance', 'High', 6.82, 'HDFC'],
        [32.19, 'Technology', 'Low', 18.46, 'WIPRO'],
        [27.83, 'Technology', 'Moderate', 13.59, 'HCLTECH'],
        [31.72, 'Technology', 'Moderate', 10.73, 'TECHM'],
        [29.65, 'Technology', 'High', 9.25, 'INFIBEAM'],
    ]
    return data

def train_random_forest_classifier(data):
    """Trains a random forest classifier."""
    data_df = pd.DataFrame(data)
    features = data_df.columns[0:3]  # Use the first three columns as features
    target_investment_type = data_df.columns[3]  # Define the target variable

    # Apply label encoding to categorical variables
    data_df[features[1]] = sector_encoder.fit_transform(data_df[features[1]])
    data_df[features[2]] = risk_type_encoder.fit_transform(data_df[features[2]])
    data_df[target_investment_type] = stock_type_encoder.fit_transform(data_df[target_investment_type])

    features_array = data_df[features].values
    target_investment_type_array = data_df[target_investment_type].values

    model = RandomForestClassifier()
    model.fit(features_array, target_investment_type_array)

    return model

investment_types_mapping = {
    0: 'Technology',
    1: 'Finance',
    2: 'Healthcare',
    3: 'Infrastructure',
    4: 'Oil & Gas',
    5: 'FMCG'
}

stock_types_mapping = {
    0: ['TCS', 'INFY', 'WIPRO', 'TECHM'],
    1: ['SBIN', 'ICICIBANK', 'BAJFINANCE'],
    2: ['HINDUNILVR', 'DRREDDY', 'CIPLA', 'DIVISLAB'],
    3: ['LT'],
    4: ['RELIANCE', 'MOBILOIL', 'ARAMCO'],
    5: ['HINDUNILVR']
}

def make_investment_recommendation(model, risk_tolerance, investment_goals, num_recommendations=3):
    """Makes investment recommendations based on the given risk tolerance and investment goals."""
    # Preprocess risk tolerance and investment goals
    risk_tolerance_values = ['Low', 'Medium', 'High']
    investment_goals_values = ['Retirement', 'Education', 'Wealth Accumulation', 'Infrastructure', 'Oil & Gas', 'FMCG']
    risk_tolerance_encoded = risk_tolerance_values.index(risk_tolerance.capitalize())
    investment_goals_encoded = investment_goals_values.index(investment_goals)

    # Generate investment recommendations
    recommendations = []
    for _ in range(num_recommendations):
        features = [
            risk_tolerance_encoded,
            random.random(),
            random.random()
        ]
        probabilities = model.predict_proba([features])[0]

        investment_type_encoded = np.argmax(probabilities)
        investment_type = investment_types_mapping.get(investment_type_encoded, 'Unknown')
        stock_type_candidates = stock_types_mapping.get(investment_type_encoded, ['Other Category'])
        stock_type = random.choice(stock_type_candidates)

        recommendation = {
            'investment_type': investment_type,
            'probability': probabilities[investment_type_encoded],
            'stock_type': stock_type
        }
        recommendations.append(recommendation)

    return recommendations




@app.route('/')
def home():
    """Render the home page."""
    return render_template('index2.html')

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    risk_tolerance = request.form['risk-tolerance']
    investment_goals = request.form['investment-goals']

    # Generate random data for training
    data = generate_random_data()

    # Train the investment model
    investment_model = train_random_forest_classifier(data)

    # Make investment recommendations
    try:
        investment_recommendations = make_investment_recommendation(
            investment_model, risk_tolerance, investment_goals, num_recommendations=3
        )
    except ValueError as e:
        return render_template('error.html', message=str(e))

    return render_template(
        'recommendations.html',
        investment_recommendations=investment_recommendations
    )

if __name__ == '__main__':
    app.run()
