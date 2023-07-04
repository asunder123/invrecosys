import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


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


def make_investment_recommendation(model, profile, num_recommendations=3):
    """Makes investment recommendations based on the given profile."""
    features = [profile['feature1'], profile['feature2']]
    probabilities = model.predict_proba([features])[0]

    recommendations = []
    sorted_indices = probabilities.argsort()[::-1]
    for i in sorted_indices[:num_recommendations]:
        probability = probabilities[i]
        risk = classify_risk(probability)
        recommendation = {
            'investment_type': i,
            'probability': probability,
            'risk': risk
        }
        recommendations.append(recommendation)

    return recommendations


def main():
    data = generate_random_data()

    model = train_random_forest_classifier(data)

    # Display model details
    print("Trained Model:")
    print(model)
    print()

    while True:
        feature1 = float(input("Enter feature 1 value: "))
        feature2 = float(input("Enter feature 2 value: "))

        profile = {
            'feature1': feature1,
            'feature2': feature2
        }

        # Make investment recommendations based on profile
        recommendations = make_investment_recommendation(model, profile, num_recommendations=4)
        print("Investment Recommendations:")
        if recommendations:
            for recommendation in recommendations:
                investment_type = recommendation['investment_type']
                probability = recommendation['probability']
                risk = recommendation['risk']
                print(f"Investment Type: {investment_type}, Probability: {probability}, Risk: {risk}")
        else:
            print("No recommendations found.")

        choice = input("Do you want to make another recommendation? (y/n): ")
        if choice.lower() != 'y':
            break


if __name__ == "__main__":
    main()
