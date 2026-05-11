# API Documentation

## Base URL
```
http://localhost:8000
```

## Interactive Documentation
FastAPI provides automatic interactive documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Endpoints

### 1. Health Check

**GET** `/`

Check if the API is running.

**Response:**
```json
{
  "message": "Disease Prediction System API",
  "version": "1.0",
  "status": "running"
}
```

---

### 2. Get Status

**GET** `/status`

Get current training status and dataset information.

**Response:**
```json
{
  "is_trained": true,
  "dataset_info": {
    "dataset_name": "Breast Cancer",
    "target": "diagnosis",
    "features": ["radius_mean", "texture_mean", ...],
    "train_shape": [455, 30],
    "val_shape": [152, 30],
    "test_shape": [152, 30]
  },
  "metrics": {
    "test_accuracy": 0.96,
    "precision": 0.95,
    ...
  }
}
```

---

### 3. Upload Dataset

**POST** `/upload`

Upload and preprocess a CSV dataset.

**Request:**
- Content-Type: `multipart/form-data`
- Body: File upload

**curl Example:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@dataset.csv"
```

**Response:**
```json
{
  "success": true,
  "message": "Dataset uploaded successfully: Breast Cancer",
  "info": {
    "dataset_name": "Breast Cancer",
    "target": "diagnosis",
    "features": ["radius_mean", "texture_mean", ...],
    "train_shape": [455, 30],
    "val_shape": [152, 30],
    "test_shape": [152, 30]
  }
}
```

**Error Response:**
```json
{
  "detail": "Unknown dataset structure. Cannot detect target column."
}
```

---

### 4. Train Model

**POST** `/train`

Train the system using GA, PSO, and DE optimization algorithms.

**Request Body:**
```json
{
  "generations": 20,
  "population": 20
}
```

**Parameters:**
- `generations` (int): Number of generations for optimization (default: 20)
- `population` (int): Population size for optimization (default: 20)

**curl Example:**
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"generations": 25, "population": 30}'
```

**Response:**
```json
{
  "success": true,
  "message": "Training completed successfully",
  "results": {
    "metrics": {
      "val_accuracy": 0.9605,
      "test_accuracy": 0.9671,
      "precision": 0.9583,
      "recall": 0.9744,
      "f1_score": 0.9663,
      "roc_auc": 0.9891,
      "confusion_matrix": [[55, 2], [2, 93]],
      "selected_features_count": 12,
      "ga_features": 15,
      "pso_features": 14,
      "de_features": 13
    },
    "convergence": {
      "ga": [0.85, 0.87, 0.89, ...],
      "pso": [0.84, 0.88, 0.91, ...],
      "de": [0.83, 0.86, 0.90, ...]
    },
    "selected_features": [
      "radius_mean",
      "texture_mean",
      "perimeter_mean",
      ...
    ]
  }
}
```

**Error Response:**
```json
{
  "detail": "No dataset uploaded"
}
```

---

### 5. Make Prediction

**POST** `/predict`

Make a prediction on a single sample.

**Request Body:**
```json
{
  "features": {
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    "perimeter_mean": 122.8,
    ...
  }
}
```

**curl Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "radius_mean": 17.99,
      "texture_mean": 10.38,
      "perimeter_mean": 122.8
    }
  }'
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "prediction": 1,
    "probability": 0.9234,
    "confidence": 0.9234,
    "voting_proba": [0.0766, 0.9234],
    "nn_prediction": 1,
    "nn_proba": [0.0821, 0.9179],
    "individual_predictions": {
      "lr": 1,
      "rf": 1,
      "svm": 1,
      "nn": 1
    }
  }
}
```

**Field Descriptions:**
- `prediction`: Final prediction (0 or 1)
- `probability`: Probability of positive class
- `confidence`: Maximum probability (confidence in prediction)
- `voting_proba`: Probabilities for each class [class_0, class_1]
- `nn_prediction`: Neural network prediction
- `nn_proba`: Neural network probabilities
- `individual_predictions`: Predictions from each model

**Error Response:**
```json
{
  "detail": "Model not trained yet"
}
```

---

### 6. Batch Prediction

**POST** `/predict_batch`

Make predictions on multiple samples at once.

**Request Body:**
```json
[
  {
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    ...
  },
  {
    "radius_mean": 13.54,
    "texture_mean": 14.36,
    ...
  }
]
```

**curl Example:**
```bash
curl -X POST "http://localhost:8000/predict_batch" \
  -H "Content-Type: application/json" \
  -d '[
    {"radius_mean": 17.99, "texture_mean": 10.38},
    {"radius_mean": 13.54, "texture_mean": 14.36}
  ]'
```

**Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "prediction": 1,
      "probability": 0.9234,
      "confidence": 0.9234,
      ...
    },
    {
      "prediction": 0,
      "probability": 0.1456,
      "confidence": 0.8544,
      ...
    }
  ],
  "count": 2
}
```

---

### 7. Add Incremental Data

**POST** `/add_data`

Add new data incrementally to the training set.

**Request:**
- Content-Type: `multipart/form-data`
- Body: CSV file with same structure as original dataset

**curl Example:**
```bash
curl -X POST "http://localhost:8000/add_data" \
  -F "file=@new_data.csv"
```

**Response:**
```json
{
  "success": true,
  "message": "Incremental data added successfully",
  "info": {
    "new_samples": 50,
    "total_samples": 505
  }
}
```

---

### 8. Get All Features

**GET** `/features`

Get list of all feature names from the dataset.

**Response:**
```json
{
  "features": [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    ...
  ],
  "count": 30
}
```

---

### 9. Get Selected Features

**GET** `/selected_features`

Get list of features selected by the ensemble after training.

**Response:**
```json
{
  "selected_features": [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "concavity_mean",
    ...
  ],
  "count": 12
}
```

**Error Response:**
```json
{
  "detail": "Model not trained yet"
}
```

---

### 10. Get Metrics

**GET** `/metrics`

Get training metrics after model training.

**Response:**
```json
{
  "val_accuracy": 0.9605,
  "test_accuracy": 0.9671,
  "precision": 0.9583,
  "recall": 0.9744,
  "f1_score": 0.9663,
  "roc_auc": 0.9891,
  "confusion_matrix": [
    [55, 2],
    [2, 93]
  ],
  "selected_features_count": 12,
  "ga_features": 15,
  "pso_features": 14,
  "de_features": 13
}
```

---

### 11. Load Saved Models

**POST** `/load_models`

Load previously saved models from disk.

**Response:**
```json
{
  "success": true,
  "message": "Models loaded successfully"
}
```

**Error Response:**
```json
{
  "detail": "Model files not found"
}
```

---

## Error Codes

| Status Code | Meaning |
|-------------|---------|
| 200 | Success |
| 400 | Bad Request (invalid input) |
| 404 | Not Found |
| 500 | Internal Server Error |

---

## Rate Limiting

Currently, there are no rate limits. For production use, consider implementing rate limiting.

---

## Authentication

Currently, no authentication is required. For production use, implement JWT or API key authentication.

---

## CORS

CORS is enabled for all origins (`*`). Modify in `app.py` for production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Python Client Example

```python
import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"

# 1. Upload dataset
with open('dataset.csv', 'rb') as f:
    files = {'file': f}
    response = requests.post(f"{BASE_URL}/upload", files=files)
    print(response.json())

# 2. Train model
config = {
    "generations": 20,
    "population": 20
}
response = requests.post(
    f"{BASE_URL}/train",
    json=config
)
print(response.json())

# 3. Make prediction
features = {
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    # ... other features
}
response = requests.post(
    f"{BASE_URL}/predict",
    json={"features": features}
)
result = response.json()
print(f"Prediction: {result['prediction']['prediction']}")
print(f"Confidence: {result['prediction']['confidence']:.2%}")
```

---

## JavaScript/Frontend Example

```javascript
// Upload dataset
async function uploadDataset(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
}

// Train model
async function trainModel(generations = 20, population = 20) {
    const response = await fetch('http://localhost:8000/train', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ generations, population })
    });
    
    return await response.json();
}

// Make prediction
async function predict(features) {
    const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ features })
    });
    
    return await response.json();
}
```

---

## Testing with Postman

1. Import the following collection or create requests manually
2. Set base URL variable: `{{base_url}}` = `http://localhost:8000`
3. Test endpoints in order:
   - GET `/status`
   - POST `/upload` (with CSV file)
   - POST `/train` (with JSON body)
   - POST `/predict` (with feature values)

---

## Monitoring

Check server logs for debugging:
- Uvicorn logs show all requests
- Check terminal output for errors
- Use `/docs` endpoint for interactive testing

---

## Performance Tips

1. **Training Time**: Depends on dataset size and parameters
   - Small dataset (500 samples): 1-2 minutes
   - Medium dataset (5000 samples): 3-5 minutes
   - Large dataset (50000 samples): 10-20 minutes

2. **Prediction Time**: Very fast
   - Single prediction: <50ms
   - Batch prediction (100 samples): <500ms

3. **Optimization**:
   - Use smaller generations/population for testing
   - Increase for production accuracy
   - Cache models after training

---

**For more details, visit the interactive documentation at `/docs` when the server is running.**
