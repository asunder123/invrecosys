import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from google.cloud import aiplatform

def train_random_forest_classifier(data):
    """Trains a random forest classifier."""
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

    data_df = pd.DataFrame(data)
    features = data_df.columns[0:3]  # Use the first three columns as features
    target_stock_type = data_df.columns[-1]  # Define the target variable

    sector_encoder = LabelEncoder()  # Initialize the sector encoder
    risk_type_encoder = LabelEncoder()

    # Apply label encoding to categorical variables
    data_df[features[1]] = sector_encoder.fit_transform(data_df[features[1]])
    data_df[features[2]] = risk_type_encoder.fit_transform(data_df[features[2]])

    features_array = data_df[features].values
    target_stock_type_array = data_df[target_stock_type].values

    model = RandomForestClassifier()
    model.fit(features_array, target_stock_type_array)

    # Get a single decision tree from the forest
    tree = model.estimators_[0]

    print(tree)

    joblib.dump(model, 'test.pkl')
    model = joblib.load('test.pkl')

    print(model)

    # Convert the model to the Vertex AI format
    project_id = 'gen-predictors'
    model_display_name = 'predict-anand-upload'
    location = 'us-central1'  # Example: 'us-central1'

    aiplatform.init(project=project_id, location=location)

    model_service_client = aiplatform.gapic.ModelServiceClient(client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"})
    model_path = model_service_client.upload_model(
        parent=f"projects/{project_id}/locations/{location}",
        model_display_name=model_display_name,
        metadata={"schema": "tensorflow_saved_model_2"},
        artifact_uri="gs://gen-predictors/predict-anand-upload"
    )

    print("Model uploaded to Vertex AI:", model_path)

    return model

train_random_forest_classifier([
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
    ])