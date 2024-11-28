import os
import pickle
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse  
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
MODEL_PATH = "water_potability_model.pkl"  
MODEL_KERAS_PATH = "water_potability_model.keras"  
DATASET_PATH = "data/water_potability.csv"

# Load initial model (pickle format)
def load_pickled_model():
    try:
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
            return model
    except FileNotFoundError:
        raise RuntimeError(f"Model file {MODEL_PATH} not found.")

# Convert pickled model to Keras format 
def convert_to_keras(model):
    if isinstance(model, tf.keras.models.Model):
        model.save(MODEL_KERAS_PATH)  # Directly save Keras model
    else:
        raise ValueError("Model is not a Keras model, cannot convert it.")

# Try loading the model or convert it to .keras format
model = load_pickled_model()
convert_to_keras(model)

# Define the schema for water sample input
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

        accuracy = accuracy_score(testY, model.predict(testX).round())
        return {"message": "Model retrained successfully", "accuracy": accuracy}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


@app.get("/download_model/")
def download_model():
    """Endpoint to download the retrained model."""
    if not os.path.exists(MODEL_KERAS_PATH):
        raise HTTPException(status_code=404, detail="Model file not found.")
    return FileResponse(MODEL_KERAS_PATH, media_type="application/octet-stream", filename="water_potability_model.keras")
