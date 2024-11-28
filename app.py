import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2


# Define FastAPI app
app = FastAPI()

# Load initial model pipeline
MODEL_PATH = "water_potability_model.pkl"

try:
    with open(MODEL_PATH, "rb") as file:
        pipeline = pickle.load(file)
except FileNotFoundError:
    raise RuntimeError(f"Model file {MODEL_PATH} not found.")

# Placeholder for dataset
DATASET_PATH = "data/water_potability.csv"  


class WaterSample(BaseModel):
    """Schema for a water sample prediction input."""
    pH: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_Carbon: float
    Trihalomethanes: float
    Turbidity: float


@app.post("/predict/")
def predict_potability(sample: WaterSample):
    """Endpoint to predict water potability."""
    try:
        input_data = pd.DataFrame([sample.dict()])
        prediction = pipeline.predict(input_data)
        potability = "Potable" if prediction[0] > 0.5 else "Not Potable"
        return {"potability": potability}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/retrain/")
def retrain_model(file: UploadFile = File(...)):
    """Endpoint to retrain the model with new data."""
    try:
        # Load new data
        new_data = pd.read_csv(file.file)

        # Check for NaN or infinite values in the dataset and log them
        if new_data.isnull().any().any():
            print("Warning: Dataset contains NaN values.")
            print(new_data[new_data.isnull().any(axis=1)])  # Log rows with NaN values

        if (new_data == float('inf')).any().any() or (new_data == float('-inf')).any().any():
            print("Warning: Dataset contains infinite values.")
            print(new_data[(new_data == float('inf')) | (new_data == float('-inf'))])  # Log rows with infinite values

        # Clean the dataset by removing rows with NaN or infinite values
        new_data = new_data.replace([float('inf'), float('-inf')], float('nan'))  # Replace infinities with NaN
        new_data = new_data.dropna()  # Drop rows with NaN values

        # Check if the dataset still contains NaN values after cleaning
        if new_data.isnull().any().any():
            raise HTTPException(status_code=400, detail="Dataset contains NaN values after cleaning. Please review the data.")

        if "Potability" not in new_data.columns:
            raise HTTPException(status_code=400, detail="Uploaded dataset must include 'Potability' column.")

        # Split into features and labels
        X = new_data.drop("Potability", axis=1)
        y = new_data["Potability"]

        # Check for NaN or infinite values in the target variable 'Potability'
        if y.isnull().any() or (y == float('inf')).any() or (y == float('-inf')).any():
            raise HTTPException(status_code=400, detail="Target variable 'Potability' contains NaN or infinite values. Please clean the data.")

        # Normalize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train/test split
        trainX, testX, trainY, testY = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Define the retrained model
        model = Sequential([
            Dense(64, activation="relu", input_shape=(trainX.shape[1],), kernel_regularizer=l2(0.001)),
            Dropout(0.4),
            Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(1, activation="sigmoid")
        ])

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

        # Training callbacks
        early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001)

        # Retrain the model
        history = model.fit(
            trainX, trainY,
            validation_data=(testX, testY),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        # Print training history for debugging
        print("Training history:", history.history)

        # Save the retrained model using Keras' built-in saving method
        model.save("retrained_model.h5")

        # Optionally, pickle the model if needed (though it's better to use `h5` format for Keras models)
        with open(MODEL_PATH, "wb") as file:
            pickle.dump(model, file)

        # Evaluate the model on the test set
        accuracy = accuracy_score(testY, model.predict(testX).round())
        
        # Return a success message with the model accuracy
        return {"message": "Model retrained successfully", "accuracy": accuracy}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")



@app.post("/trigger-retrain/")
def trigger_retraining():
    """Manually trigger retraining."""
    # Load dataset for retraining
    try:
        data = pd.read_csv(DATASET_PATH)
        return retrain_model(file=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining trigger failed: {str(e)}")
