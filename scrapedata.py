import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the dataset from the provided input
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

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data, columns=['Value', 'Sector', 'Risk', 'Rating', 'Symbol'])

# Perform data preprocessing and feature engineering as needed
# ...

# Encode categorical variables using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Sector', 'Risk'])

# Split the data into features (X) and labels (y)
X = df_encoded.drop(['Symbol'], axis=1)
y = df_encoded['Symbol']

# Create a machine learning model using TensorFlow and Keras
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(df_encoded['Symbol'].unique()), activation='softmax')
])

# Compile the model with an appropriate optimizer, loss function, and metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10)

# Save the trained model
model.save('investment_model')

# Print a summary of the trained model
model.summary()
