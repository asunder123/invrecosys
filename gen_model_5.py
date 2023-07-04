import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request

app = Flask(__name__)

# Initialize the label encoders
risk_tolerance_encoder = LabelEncoder()
investment_goals_encoder = LabelEncoder()
stock_type_encoder = LabelEncoder()


def generate_random_data():
    """Generates random data for training."""
    data = []
    for _ in range(10000):
        features = [random.random(), random.random()]
        risk = random.choice(['Low', 'Medium', 'High'])
        stock_type = random.choice(['Technology', 'Finance', 'Healthcare'])
        data.append(features + [risk, stock_type])

    return data


def train_random_forest_classifier(data):
    """Trains a random forest classifier."""
    data_df = pd.DataFrame(data)
    features = data_df.columns[:-2]
    target_risk = data_df.columns[-2]
    target_stock_type = data_df.columns[-1]

    features_array = data_df[features].values
    target_risk_array = data_df[target_risk].values
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
    risk_tolerance_encoder.fit(risk_tolerance_values)
    investment_goals_encoder.fit(investment_goals_values)

    # Capitalize the risk_tolerance value if necessary
    risk_tolerance = risk_tolerance.capitalize()

    # Check if risk_tolerance and investment_goals are valid labels
    if risk_tolerance not in risk_tolerance_encoder.classes_:
        raise ValueError(f"Invalid risk tolerance label: {risk_tolerance}")
    if investment_goals not in investment_goals_encoder.classes_:
        raise ValueError(f"Invalid investment goals label: {investment_goals}")

    # Encode risk tolerance and investment goals
    risk_tolerance_encoded = risk_tolerance_encoder.transform([risk_tolerance])[0]
    investment_goals_encoded = investment_goals_encoder.transform([investment_goals])[0]

    # Generate investment recommendations
    recommendations = []
    for _ in range(num_recommendations):
        features = [risk_tolerance_encoded, investment_goals_encoded]
        probabilities = model.predict_proba([features])[0]
        investment_types = risk_tolerance_encoder.inverse_transform(range(len(probabilities)))

        recommendation = {
            'investment_type': random.choice(investment_types),
            'probability': random.choice(probabilities)
        }
        recommendations.append(recommendation)

    return recommendations


def make_stock_recommendation(model, num_recommendations=3):
    """Makes stock recommendations."""
    stock_types = ['Technology', 'Finance', 'Healthcare']

    recommendations = []
    for _ in range(num_recommendations):
        features = [random.random(), random.random()]
        stock_type = model.predict([features])[0]

        recommendation = {
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

    return render_template('recommendations2.html', investment_recommendations=investment_recommendations, stock_recommendations=stock_recommendations)


if __name__ == '__main__':
    app.run()
