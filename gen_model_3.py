import random
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)


def generate_random_data():
    """Generates random data for training."""
    data = []
    for _ in range(10000):
        features = [random.random(), random.random()]
        target = random.randint(0,1)
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


def preprocess_data(data):
    """Preprocesses the data by encoding categorical features."""
    encoder = LabelEncoder()
    encoded_data = data.copy()
    encoded_data['risk_tolerance'] = encoder.fit_transform(data['risk_tolerance'])
    # Add similar preprocessing steps for other categorical features if applicable
    return encoded_data


def make_investment_recommendation(model, risk_tolerance, investment_goals, num_recommendations=3):
    """Makes investment recommendations based on the given risk tolerance and investment goals."""
    profile = {
        'risk_tolerance': risk_tolerance,
        'investment_goals': investment_goals
    }

    # Preprocess the profile data
    profile_data = pd.DataFrame([profile])
    profile_data = preprocess_data(profile_data)

    # Get the features for the profile
    features = profile_data.values[0]

    # Use the trained model to predict probabilities
    probabilities = model.predict_proba([features])[0]

    # Generate recommendations based on probabilities
    recommendations = []
    for i, probability in enumerate(probabilities):
        investment_type = i
        risk = classify_risk(probability)
        recommendation = {
            'investment_type': investment_type,
            'probability': probability,
            'risk': risk
        }
        recommendations.append(recommendation)

    # Sort recommendations by probability (descending)
    recommendations = sorted(recommendations, key=lambda x: x['probability'], reverse=True)

    # Select the top num_recommendations
    recommendations = recommendations[:num_recommendations]

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
