import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 1. Initialize FastAPI app
app = FastAPI(title="House Price Prediction API", version="1.0.0")

# 2. Enable CORS so your HTML/JS page can make requests safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your exact frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "House Price Prediction API is up and running smoothly!"}

# 3. Load the pre-trained Scikit-Learn Pipeline
# 3. Load the pre-trained Scikit-Learn Pipeline
MODEL_PATH = "house_price_model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    
    # Extract the actual Scikit-Learn Pipeline object from the dictionary
    model_pipeline = model_data["model"] [cite: 4]
    
    print("Model pipeline successfully extracted from container!")
except Exception as e:
    raise RuntimeError(f"Failed to load model file at {MODEL_PATH}: {str(e)}")


# 4. Define the input data schema matching model parameters
class HouseFeatures(BaseModel):
    tsp_en: str = Field(..., description="Township name (e.g., Bahan, South Okkalapa)")
    bedroom: float = Field(..., ge=0)
    bathroom: float = Field(..., ge=0)
    aircorn: float = Field(..., ge=0, description="Air conditioner count or configuration status")
    floor: float = Field(..., description="Floor number level")
    overhead_tank: float = Field(..., description="Overhead tank availability metric")

@app.post("/predict")
def predict_price(data: HouseFeatures):
    try:
        # Convert incoming JSON payload to a pandas DataFrame structured for the pipeline
        input_df = pd.DataFrame([data.model_dump()])
        
        # Execute the pipeline transformation stages and output inference
        prediction = model_pipeline.predict(input_df)
        
        # Extract scalar value from numpy output array
        predicted_price = float(prediction[0])
        
        return {
            "status": "success",
            "predicted_price": round(predicted_price, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")

@app.get("/townships")
def get_supported_townships():
    """Returns valid categorical variables parsed from model attributes."""
    return {
        "townships": [
            "Ahlone", "Bahan", "Botahtaung", "Dagon Myothit (North)", 
            "Dagon Myothit (South)", "Dawbon", "Hlaing", "Insein", 
            "Kamaryut", "Kyauktada", "Kyeemyindaing", "Lanmadaw", 
            "Mayangone", "Mingalartaungnyunt", "North Okkalapa", 
            "Pazundaung", "Sanchaung", "South Okkalapa", "Tamwe", 
            "Thaketa", "Thingangyun", "Yankin"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)