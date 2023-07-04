import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request

app = Flask(__name__)

# Initialize the label encoder and fit it with the encoded values
encoder = LabelEncoder()


def generate_random_data():
    """Generates random data for training."""
    data = []
    for _ in range(10000):
        features = [random.random(), random.random(),random.random()]
        risk = random.choice(['Low', 'Medium', 'High'])
        data.append(features + [risk])

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
    encoder = LabelEncoder()
    encoder.fit(risk_tolerance_values + investment_goals_values)

    # Capitalize the risk_tolerance value if necessary
    risk_tolerance = risk_tolerance.capitalize()

    # Check if risk_tolerance and investment_goals are valid labels
    if risk_tolerance not in encoder.classes_:
        raise ValueError(f"Invalid risk tolerance label: {risk_tolerance}")
    if investment_goals not in encoder.classes_:
        raise ValueError(f"Invalid investment goals label: {investment_goals}")

    # Encode risk tolerance and investment goals
    risk_tolerance_encoded = encoder.transform([risk_tolerance])[0]
    investment_goals_encoded = encoder.transform([investment_goals])[0]

    # Generate investment recommendations
    recommendations = []
    for _ in range(num_recommendations):
        features = [risk_tolerance_encoded, investment_goals_encoded]
        probabilities = model.predict_proba([features])[0]
        investment_types = encoder.inverse_transform(range(len(probabilities)))

        recommendation = {
            'investment_type': random.choice(investment_types),
            
            'probability': random.choice(probabilities)
        }
        recommendations.append(recommendation)

    return recommendations


@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')


@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    """Handle the form submission and return the recommendations."""
    risk_tolerance = request.form['risk-tolerance']
    investment_goals = request.form['investment-goals']

    # Generate random data for training
    data = generate_random_data()

    # Train the model
    model = train_random_forest_classifier(data)

    # Make investment recommendations
    try:
        recommendations = make_investment_recommendation(model, risk_tolerance, investment_goals, num_recommendations=3)
    except ValueError as e:
        return render_template('error.html', message=str(e))

    return render_template('recommendations.html', recommendations=recommendations)


if __name__ == '__main__':
    app.run()
