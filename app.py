import os
import pickle
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse  
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from starlette.middleware.cors import CORSMiddleware

# Define FastAPI app
app = FastAPI()

# Allow cross-origin requests from localhost (or your frontend origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Paths to model files and dataset
MODEL_KERAS_PATH = "water_potability_model.keras"  
DATASET_PATH = "data/water_potability.csv"

# Load the Keras model if exists
if os.path.exists(MODEL_KERAS_PATH):
    model = tf.keras.models.load_model(MODEL_KERAS_PATH)
else:
    model = None

# Define the schema for water sample input
class WaterSample(BaseModel):
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
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try:
        input_data = pd.DataFrame([sample.dict()])
        prediction = model.predict(input_data)
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
        if "Potability" not in new_data.columns:
            raise HTTPException(status_code=400, detail="Uploaded dataset must include 'Potability' column.")

        # Split into features and labels
        X = new_data.drop("Potability", axis=1)
        y = new_data["Potability"]

        # Handle missing values (Fill with the mean of each column)
        X = X.fillna(X.mean())

        # Train/test split
        trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)

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
        model.fit(trainX, trainY, validation_data=(testX, testY), epochs=30, batch_size=32, callbacks=[early_stopping, reduce_lr], verbose=1)

        # Save the retrained model in .keras format
        model.save(MODEL_KERAS_PATH)  

        # Model evaluation
        predictions = model.predict(testX).round()
        accuracy = accuracy_score(testY, predictions)
        precision = precision_score(testY, predictions, zero_division=0)
        recall = recall_score(testY, predictions, zero_division=0)
        f1 = f1_score(testY, predictions, zero_division=0)
        loss = model.evaluate(testX, testY, verbose=0)[0]

        return {
            "message": "Model retrained successfully",
            "accuracy": accuracy,
            "loss": loss,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")
