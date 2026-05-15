# FastAPI backend (ML logic)
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import io
import json

# Import ML utilities
from model_utils import (
    preprocess_dataset, train_models, make_prediction,
    save_models, load_saved_models, add_new_data,
    get_selected_features, FEATURE_NAMES
)

app = FastAPI(title="Disease Prediction System API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
training_status = {"is_trained": False, "dataset_info": None}


# ============================================
# Pydantic Models
# ============================================

class PredictionInput(BaseModel):
    features: Dict[str, float]


class TrainingConfig(BaseModel):
    generations: int = 20
    population: int = 20


class IncrementalDataInput(BaseModel):
    data: List[Dict[str, float]]


# ============================================
# API Endpoints
# ============================================

@app.get("/")
def read_root():
    return {
        "message": "Disease Prediction System API",
        "version": "1.0",
        "status": "running"
    }


@app.get("/status")
def get_status():
    """Get training status and dataset info"""
    return training_status


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload and preprocess dataset
    Supports: Breast Cancer, Heart Disease, Diabetes datasets
    """
    global training_status
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Preprocess dataset
        info = preprocess_dataset(df, file.filename)
        
        training_status["dataset_info"] = info
        training_status["is_trained"] = False
        
        return {
            "success": True,
            "message": f"Dataset uploaded successfully: {info['dataset_name']}",
            "info": info
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/train")
async def train_system(config: TrainingConfig):
    """
    Train the system using GA/PSO/DE optimization
    Returns convergence history and metrics
    """
    global training_status
    
    if training_status["dataset_info"] is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded")
    
    try:
        # Train models
        results = train_models(gens=config.generations, pop=config.population)
        
        # Save models
        save_models()
        
        training_status["is_trained"] = True
        training_status["metrics"] = results["metrics"]
        training_status["selected_features"] = results["selected_features"]
        training_status["convergence"] = results["convergence"]
        training_status["ga_metrics"] = results.get("ga_metrics")
        training_status["pso_metrics"] = results.get("pso_metrics")
        training_status["de_metrics"] = results.get("de_metrics")
        
        return {
            "success": True,
            "message": "Training completed successfully",
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(data: PredictionInput):
    """
    Make prediction on new unseen data
    Returns prediction, probability, and confidence
    """
    if not training_status["is_trained"]:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    try:
        result = make_prediction(data.features)
        return {
            "success": True,
            "prediction": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
def predict_batch(data: List[Dict[str, float]]):
    """
    Make predictions on multiple samples
    """
    if not training_status["is_trained"]:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    try:
        results = []
        for sample in data:
            result = make_prediction(sample)
            results.append(result)
        
        return {
            "success": True,
            "predictions": results,
            "count": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add_data")
async def add_incremental_data(file: UploadFile = File(...)):
    """
    Add new data incrementally to the training set
    """
    if not training_status["is_trained"]:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    try:
        contents = await file.read()
        new_df = pd.read_csv(io.BytesIO(contents))
        
        result = add_new_data(new_df)
        
        return {
            "success": True,
            "message": "Incremental data added successfully",
            "info": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/features")
def get_features():
    """Get all feature names"""
    return {
        "features": FEATURE_NAMES,
        "count": len(FEATURE_NAMES)
    }


@app.get("/selected_features")
def get_selected_features_endpoint():
    """Get selected features after training"""
    if not training_status["is_trained"]:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    return {
        "selected_features": get_selected_features(),
        "count": len(get_selected_features())
    }


@app.get("/metrics")
def get_metrics():
    """Get training metrics"""
    if not training_status["is_trained"]:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    return training_status.get("metrics", {})


@app.get("/results")
def get_results():
    """Get complete training results including convergence data"""
    if not training_status["is_trained"]:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    return {
        "metrics": training_status.get("metrics", {}),
        "convergence": training_status.get("convergence", {}),
        "selected_features": training_status.get("selected_features", []),
        "ga_metrics": training_status.get("ga_metrics", {}),
        "pso_metrics": training_status.get("pso_metrics", {}),
        "de_metrics": training_status.get("de_metrics", {})
    }


@app.post("/load_models")
def load_models():
    """Load previously saved models"""
    global training_status
    
    try:
        load_saved_models()
        training_status["is_trained"] = True
        
        return {
            "success": True,
            "message": "Models loaded successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
