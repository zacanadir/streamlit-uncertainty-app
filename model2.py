import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np


# Define the model architecture for 4 classes
def create_model(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(4, activation='softmax')(x)
    model=Model(inputs=inputs,outputs=outputs)
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Function to apply MC Dropout during inference
def predict_with_mc_dropout(model, X, n_iterations=50):
    """
    Perform MC Dropout by making multiple predictions with dropout layers active.
    Arguments:
        model: The Keras model with dropout layers.
        X: Input data for prediction.
        n_iterations: Number of forward passes to simulate dropout.
    Returns:
        Mean of predictions from multiple forward passes.
    """
    # Initialize an array to hold the predictions for all iterations
    n_samples = X.shape[0]
    predictions = np.zeros((n_iterations, n_samples, 4))  # Shape: (iterations, samples, classes)

    for i in range(n_iterations):
        # Perform prediction with dropout enabled (training=True)
        # model(X, training=True) will output shape (n_samples, n_classes)
        preds = model(X, training=True).numpy()  # Shape: (n_samples, 4)
        predictions[i] = preds  # Store the prediction for this iteration

    # Average the predictions over all iterations for each sample
    return predictions.mean(axis=0),predictions.std(axis=0)  # Shape: (n_samples, 4)


# Train and save the model
def train_and_save_model(X, y, model_path='models/model2.h5'):
    model = create_model(X.shape[1])
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

    # Save the trained model in H5 format
    model.save(model_path)  # Save model in H5 format


# Load the trained model
def load_mc_dropout_model(model_path='models/model2.h5'):
    return tf.keras.models.load_model(model_path)

