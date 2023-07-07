import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request

app = Flask(__name__)

#Initialize encoders
sector_encoder = LabelEncoder()
risk_type_encoder = LabelEncoder()
stock_type_encoder = LabelEncoder()



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

"""def train_random_forest_classifier(data):

    data_df = pd.DataFrame(data)
    features = data_df.columns[0:3]  # Use the first three columns as features
    #target_stock_type = data_df.columns[-1]  # Define the target variable
    target_stock_type = data_df.columns[-1]

    # Apply label encoding to categorical variables
    data_df[features[1]] = sector_encoder.fit_transform(data_df[features[1]])
    data_df[features[2]] = risk_type_encoder.fit_transform(data_df[features[2]])

    features_array = data_df[features].values
    target_stock_type_array = data_df[target_stock_type].values

    target_stock_type_encoder = LabelEncoder()
    target_stock_type_encoded = target_stock_type_encoder.fit_transform(target_stock_type_array)

    model = RandomForestClassifier()
    model.fit(features_array, target_stock_type_encoded)

    return model"""

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

"""def make_stock_recommendation(model, num_recommendations=3):
    random_data = [[random.random() for _ in range(3)] for _ in range(num_recommendations)]
    stock_type_labels = model.predict(random_data).tolist()
    stock_types = target_stock_type_encoder.inverse_transform(stock_type_labels)

    recommendations = []
    for stock_type in stock_types:
        recommendation = {
            'stock_type': stock_type if stock_type else '',
            'other_details': f'Detail {random.randint(1, 5)}' if stock_type else ''
        }
        recommendations.append(recommendation)
    return recommendations"""

"""def make_stock_recommendation(model, num_recommendations=5):

    #stock_types = model.classes_  # Get the stock types from the model
    features=model.predict([random.random() for _ in range(3)])
    stock_types = model.predict([features]).reshape(-1,1)

    recommendations = []
    for _ in range(num_recommendations):
        features = [random.random() for _ in range(3)]
        stock_type = random.choice(stock_types)  # Choose a random stock type from the available options

        recommendation = {
            'investment_type': 'N/A',
            'probability': 0.0,
            'stock_type': stock_type,
            'other_details': 'Other details related to the stock recommendation'  # Add other details here
        }
        recommendations.append(recommendation)

    return recommendations"""



"""def make_stock_recommendation(model, num_recommendations=3):
    #stock_types = model.classes_  # Get the stock types from the model

    random_data = [[random.random() for _ in range(3)] for _ in range(num_recommendations)]
    predicted_labels = model.predict(random_data)
    stock_types = stock_type_encoder.inverse_transform(predicted_labels)

    recommendations = []
    for _ in range(num_recommendations):
        features = [random.random() for _ in range(3)]
        stock_type = random.choice(stock_types)  # Choose a random stock type from the available options

        recommendation = {
            'investment_type': 'N/A',
            'probability': 0.0,
            'stock_type': stock_type,
            'other_details': 'Other details related to the stock recommendation'  # Add other details here
        }
        recommendations.append(recommendation)

    return recommendations"""

    
def make_stock_recommendation(data, num_recommendations=3):
    recommendations = []
    for _ in range(num_recommendations):
        random_index = random.randint(0, len(data) - 1)
        recommendation = {
            'investment_type': 'N/A',
            'probability': 0.0,
            'stock_type': data[random_index][-1],
            'other_details': 'Other details related to the stock recommendation'
        }
        recommendations.append(recommendation)

    return recommendations





@app.route('/')
def home():
    """Render the home page."""
    return render_template('index2.html')

"""@app.route('/recommendations', methods=['POST'])
def get_recommendations():

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

    return render_template('recommendations3.html', investment_recommendations=investment_recommendations, stock_recommendations=stock_recommendations)"""

"""
@app.route('/recommendations', methods=['POST'])
def get_recommendations():
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
        investment_recommendations = make_investment_recommendation(
            investment_model, risk_tolerance, investment_goals, num_recommendations=3
        )
    except ValueError as e:
        return render_template('error.html', message=str(e))

    # Make stock recommendations
    stock_recommendations = make_stock_recommendation(stock_model, num_recommendations=3)

    return render_template(
        'recommendations.html',
        investment_recommendations=investment_recommendations,
        stock_recommendations=stock_recommendations
    )"""

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
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
        investment_recommendations = make_investment_recommendation(
            investment_model, risk_tolerance, investment_goals, num_recommendations=3
        )
    except ValueError as e:
        return render_template('error.html', message=str(e))

    # Make stock recommendations
    stock_recommendations = make_stock_recommendation(data, num_recommendations=3)

    return render_template(
        'recommendations.html',
        investment_recommendations=investment_recommendations,
        stock_recommendations=stock_recommendations
    )

if __name__ == '__main__':
    app.run()
