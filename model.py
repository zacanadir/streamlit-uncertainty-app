from preprocess import load_data, get_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
import os


def build_pipeline(preprocessor):
    return Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier())
    ])

def get_train_test_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Load data and prepare
data = load_data()
X = data.drop(columns='price_range')
y = data['price_range']

# Preprocessing
preprocessor = get_preprocessor()

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model pipeline
pipeline = build_pipeline(preprocessor)

# Fit the model
pipeline.fit(X_train, y_train)

# Save the trained model
joblib.dump(pipeline, 'models/model.pkl')