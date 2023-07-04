import random
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier


app = Flask(__name__)


def generate_random_data():
    """Generates random data for training."""
    data = []
    for _ in range(100):
        features = [random.random(), random.random()]
        target = random.randint(0, 1)
        data.append(features + [target])

    return data


def train_random_forest_classifier(data):
    """Trains a random forest classifier."""
    data_df = pd.DataFrame(data)
    features = data_df.columns[:-1]
    target = data_df.columns[-1]

    features_array = data_df[features].values
    target_array = data_df[target].values

    model = RandomForestClassifier()
    model.fit(features_array, target_array)

    return model


def classify_risk(probability):
    """Classifies the investment recommendation based on probability."""
    if probability >= 0.8:
        return "Aggressive"
    elif probability >= 0.6:
        return "Medium Risk"
    else:
        return "Low Risk"


def make_investment_recommendation(model, risk_tolerance, investment_goals, num_recommendations=3):
    """Makes investment recommendations based on the given risk tolerance and investment goals."""
    profile = {
        'risk_tolerance': risk_tolerance,
        'investment_goals': investment_goals
    }

    # Replace with your recommendation logic based on risk tolerance and investment goals
    # Here, a random recommendation is generated for demonstration purposes
    recommendations = []
    for _ in range(num_recommendations):
        investment_type = random.choice(['Stocks', 'Bonds', 'Real Estate'])
        probability = random.random()
        risk = classify_risk(probability)
        recommendation = {
            'investment_type': investment_type,
            'probability': probability,
            'risk': risk
        }
        recommendations.append(recommendation)

    return recommendations


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    risk_tolerance = request.form['risk-tolerance']
    investment_goals = request.form['investment-goals']

    # Load or generate the training data
    data = generate_random_data()

    # Train the model
    model = train_random_forest_classifier(data)

    # Generate investment recommendations
    recommendations = make_investment_recommendation(model, risk_tolerance, investment_goals, num_recommendations=4)

    return jsonify(recommendations)


if __name__ == '__main__':
    app.run(debug=True)
