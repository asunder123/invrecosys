import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request

app = Flask(__name__)

# Initialize the label encoders
sector_encoder = LabelEncoder()
risk_type_encoder = LabelEncoder()

def generate_random_data():
    """Generates random data for training."""
    data = [
        [28.52, 'Oil & Gas - Refining & Marketing', 'Moderate', 14.37, 'RELIANCE'],
        [29.84, 'IT Services', 'Low', 21.13, 'TCS'],
        [30.67, 'Banking', 'Low', 16.43, 'HDFCBANK'],
        [29.11, 'IT Services', 'Moderate', 18.69, 'INFY'],
        [24.85, 'Banking', 'Moderate', 11.76, 'ICICIBANK'],
        [11.73, 'Banking', 'High', 8.52, 'SBIN'],
        [31.92, 'Banking', 'Moderate', 14.48, 'KOTAKBANK'],
        [67.92, 'FMCG', 'Low', 4.63, 'HINDUNILVR'],
        [21.27, 'Infrastructure', 'Moderate', 9.32, 'LT'],
        [25.56, 'Banking', 'Moderate', 12.52, 'AXISBANK']
    ]

    # Append random data
    #for _ in range(100):
    #    features = [
    #        random.random() * 100,
    #        random.choice(['Sector 1', 'Sector 2', 'Sector 3']),
    #        random.choice(['Low', 'Medium', 'High']),
    #        random.random() * 100,
    #        random.choice(['Technology', 'Finance', 'Healthcare'])
    #    ]
    #    data.append(features)

    return data

def train_random_forest_classifier(data):
    """Trains a random forest classifier."""
    data_df = pd.DataFrame(data)
    features = data_df.columns[0:3]  # Use the first three columns as features
    target_stock_type = data_df.columns[-1]  # Define the target variable

    # Apply label encoding to categorical variables
    data_df[features[1]] = sector_encoder.fit_transform(data_df[features[1]])
    data_df[features[2]] = risk_type_encoder.fit_transform(data_df[features[2]])

    features_array = data_df[features].values
    target_stock_type_array = data_df[target_stock_type].values

    model = RandomForestClassifier()
    model.fit(features_array, target_stock_type_array)

    return model

def classify_risk(probability):
    """Classifies the investment recommendation based on probability."""
    if probability >= 0.8:
        return "High Risk"
    elif probability >= 0.6:
        return "Medium Risk"
    else:
        return "Low Risk"

def make_investment_recommendation(model, risk_tolerance, investment_goals, num_recommendations=3):
    """Makes investment recommendations based on the given risk tolerance and investment goals."""
    # Preprocess risk tolerance and investment goals
    risk_tolerance_values = ['Low', 'Medium', 'High']
    investment_goals_values = ['Retirement', 'Education', 'Wealth Accumulation']
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
        investment_type = random.choice(['Technology', 'Finance', 'Healthcare'])

        recommendation = {
            'investment_type': investment_type,
            'probability': probabilities[investment_goals_encoded]
        }
        recommendations.append(recommendation)

    return recommendations

def make_stock_recommendation(model, num_recommendations=5):
    """Makes stock recommendations."""
    stock_types = model.classes_  # Get the stock types from the model

    recommendations = []
    for _ in range(num_recommendations):
        features = [random.random() for _ in range(3)]
        stock_type = random.choice(stock_types)  # Choose a random stock type from the available options
        stock_performance = random.random() * 100  # Generate a random stock performance value

        recommendation = {
            'investment_type': 'N/A',
            'probability': 0.0,
            'stock_type': stock_type,
            'other_details': 'Other details related to the stock recommendation',
            'stock_performance': stock_performance
        }
        recommendations.append(recommendation)

    return recommendations




@app.route('/')
def home():
    """Render the home page."""
    return render_template('index2.html')

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    """Handle the form submission and return the recommendations."""
    risk_tolerance = request.form['risk-tolerance']
    investment_goals = request.form['investment-goals']

    # Generate random data for training
    data = generate_random_data()

    # Train the investment model
    investment_model = train_random_forest_classifier(data)

    # Train the stock model
    stock_model = train_random_forest_classifier(data)

    # Make investment recommendations
    try:
        investment_recommendations = make_investment_recommendation(investment_model, risk_tolerance, investment_goals, num_recommendations=3)
    except ValueError as e:
        return render_template('error.html', message=str(e))

    # Make stock recommendations
    stock_recommendations = make_stock_recommendation(stock_model, num_recommendations=3)

    # Generate random probability values for demonstration purposes
    probability_values = [random.random() for _ in range(3)]

    # Get stock performance values
    stock_performance_values = [recommendation['stock_performance'] for recommendation in stock_recommendations]

    return render_template('recommendations.html', investment_recommendations=investment_recommendations, stock_recommendations=stock_recommendations, probability_values=probability_values, stock_performance_values=stock_performance_values)


if __name__ == '__main__':
    app.run()