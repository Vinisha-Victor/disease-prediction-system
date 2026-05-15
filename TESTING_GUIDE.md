# Testing & Verification Guide

## Overview
This document provides step-by-step instructions to test all features of the Disease Prediction System.

## Prerequisites
1. Python 3.8+ installed
2. All dependencies installed (`pip install -r backend/requirements.txt`)
3. Backend server running (`python backend/app.py`)
4. Frontend accessible (open `frontend/index.html` in browser)

---

## Test 1: Data Upload

### Steps:
1. Open `http://localhost:8000/docs` (FastAPI docs)
2. Or open `frontend/index.html` in browser
3. Click "Browse" under Data Upload section
4. Select one of the supported datasets (breast_cancer.csv, heart.csv, or diabetes.csv)
5. Click "Upload Dataset"

### Expected Results:
- ✅ Green success message: "Dataset uploaded successfully"
- ✅ Dataset info displayed: shape, features, class distribution
- ✅ Training section becomes visible

---

## Test 2: Model Training

### Steps:
1. After successful upload, scroll to Training Configuration
2. Set parameters:
   - Generations: 20 (default)
   - Population Size: 30 (default)
3. Click "Start Training"
4. Wait for training to complete (~30-60 seconds)

### Expected Results:
- ✅ Training progress messages appear
- ✅ **Individual Algorithm Results Display**:
   - **GA Card**: Shows selected features count, accuracy, F1 score
   - **PSO Card**: Shows selected features count, accuracy, F1 score
   - **DE Card**: Shows selected features count, accuracy, F1 score
- ✅ **Ensemble Metrics Display**:
   - Accuracy, Precision, Recall, F1 Score, ROC AUC
- ✅ **Confusion Matrix Generated**:
   - 2x2 grid with labeled cells
   - True Negative (top-left, green border)
   - False Positive (top-right, red border)
   - False Negative (bottom-left, red border)
   - True Positive (bottom-right, green border)
   - Actual counts displayed in each cell
- ✅ **Convergence Charts Section Visible**:
   - Combined chart showing all three algorithms (GA, PSO, DE)
   - Toggle buttons: Combined, GA, PSO, DE
   - X-axis: Generation, Y-axis: Fitness Score
- ✅ Selected features list displayed

### Verification Points:
```javascript
// In browser console, check:
console.log(trainingResults); 
// Should show:
// {
//   metrics: {...},        // Ensemble metrics
//   ga_metrics: {...},     // GA-specific metrics
//   pso_metrics: {...},    // PSO-specific metrics
//   de_metrics: {...},     // DE-specific metrics
//   convergence: {ga: [...], pso: [...], de: [...]},
//   selected_features: [...]
// }
```

---

## Test 3: Single Prediction

### Steps:
1. After training completes, scroll to Single Data Prediction
2. Enter feature values manually (or use test values)
3. Click "Predict"

### Expected Results:
- ✅ Prediction displayed: "Positive" (red) or "Negative" (green)
- ✅ Confidence percentage shown
- ✅ Probability scores displayed

### Example Test Input (Breast Cancer):
```
17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,
1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,
0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189
```

---

## Test 4: Batch Prediction

### Steps:
1. Use the provided `test_batch.csv` file
2. Click "Browse" under Batch Prediction section
3. Select `test_batch.csv`
4. Click "Predict Batch"

### Expected Results:
- ✅ No 422 error (fixed with better error handling)
- ✅ Table displayed with Sample #, Prediction, Confidence
- ✅ Color-coded predictions (red for positive, green for negative)
- ✅ Green background container

### If 422 Error Occurs:
- Check browser console for detailed error message
- Verify CSV format matches training data features
- Ensure no missing values in CSV

---

## Test 5: Convergence Chart Toggles

### Steps:
1. After training, locate the convergence charts section
2. Click each toggle button: Combined, GA, PSO, DE
3. Observe chart updates

### Expected Results:
- ✅ **Combined View**: All three algorithm lines visible
- ✅ **GA View**: Only red line (Genetic Algorithm)
- ✅ **PSO View**: Only blue line (Particle Swarm Optimization)
- ✅ **DE View**: Only teal line (Differential Evolution)
- ✅ Smooth line rendering with tension: 0.4
- ✅ Y-axis labeled "Fitness Score"
- ✅ X-axis labeled "Generation"

---

## Test 6: Algorithm Metrics Verification

### Manual Verification:
1. After training, note the three algorithm cards
2. Compare metrics:
   - Each algorithm should show different feature counts
   - Accuracy values may vary slightly
   - F1 scores reflect individual algorithm performance

### Validation:
```python
# Backend verification (check backend console logs)
# Look for output like:
# GA selected 12 features with accuracy: 0.94
# PSO selected 10 features with accuracy: 0.93
# DE selected 11 features with accuracy: 0.95
```

---

## Test 7: Confusion Matrix Validation

### Visual Checks:
1. Confirm 2x2 grid structure
2. Labels present: "True Negative", "False Positive", "False Negative", "True Positive"
3. Numbers displayed in each cell
4. Color coding:
   - Green borders: True Negative, True Positive
   - Red borders: False Positive, False Negative

### Mathematical Validation:
```
Sum of all cells should equal total test samples
True Negative + False Positive + False Negative + True Positive = Test Set Size
```

---

## Test 8: API Endpoint Testing

### Using FastAPI Docs (http://localhost:8000/docs):

#### 1. Upload Dataset
```json
POST /upload
Content-Type: multipart/form-data
file: [select breast_cancer.csv]
```

#### 2. Check Status
```json
GET /status
Response: {"is_trained": true, "dataset_loaded": true, "dataset_info": {...}}
```

#### 3. Train Model
```json
POST /train
{
  "generations": 20,
  "population_size": 30,
  "acc_weight": 1.0,
  "feature_penalty": 0.02,
  "stability_weight": 0.08
}
```

#### 4. Single Prediction
```json
POST /predict
{
  "features": {
    "mean radius": 17.99,
    "mean texture": 10.38,
    ... (all 30 features)
  }
}
```

#### 5. Batch Prediction
```json
POST /predict_batch
[
  {"mean radius": 17.99, ...},
  {"mean radius": 20.57, ...}
]
```

---

## Common Issues & Solutions

### Issue 1: 422 Unprocessable Entity on Batch Prediction
**Cause**: CSV features don't match training data features
**Solution**: 
- Ensure CSV header names exactly match training features
- Check for missing columns
- Verify feature order (doesn't matter, but names must match)

### Issue 2: Confusion Matrix Not Displaying
**Status**: ✅ FIXED
- Updated HTML structure with proper grid
- Added CSS styling with borders
- Fixed JavaScript to populate cells correctly

### Issue 3: Missing Individual Algorithm Metrics
**Status**: ✅ FIXED
- Added `evaluate_mask_metrics()` function
- Modified `train_models()` to return ga_metrics, pso_metrics, de_metrics
- Frontend displays three separate cards

### Issue 4: Graphs Not Loading
**Check**:
1. Chart.js loaded: `<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>`
2. Canvas element exists: `<canvas id="convergenceChart"></canvas>`
3. Data structure correct: `{ga: [...], pso: [...], de: [...]}`
4. No console errors

### Issue 5: CORS Error
**Solution**: Backend already has CORS middleware enabled:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
```

---

## Performance Benchmarks

### Expected Training Times:
- **Breast Cancer** (569 samples, 30 features): ~30-45 seconds
- **Heart Disease** (303 samples, 13 features): ~20-30 seconds
- **Diabetes** (768 samples, 8 features): ~15-25 seconds

### Expected Accuracy Ranges:
- **Breast Cancer**: 93-97%
- **Heart Disease**: 80-88%
- **Diabetes**: 75-82%

---

## Feature Verification Checklist

### ML Features:
- [x] Genetic Algorithm implementation
- [x] Particle Swarm Optimization implementation
- [x] Differential Evolution implementation
- [x] Dynamic fitness function with weights
- [x] Stability scoring
- [x] Incremental evaluation
- [x] Neural Network (MLPClassifier)
- [x] Ensemble voting (LR, RF, SVM)
- [x] Cross-validation
- [x] Feature selection

### Frontend Features:
- [x] Dataset upload with validation
- [x] Training configuration panel
- [x] Real-time training status
- [x] Individual algorithm metrics (GA, PSO, DE)
- [x] Ensemble metrics display
- [x] Confusion matrix visualization
- [x] Convergence charts with toggle
- [x] Single prediction form
- [x] Batch prediction CSV upload
- [x] Responsive design
- [x] Error handling

### Backend Features:
- [x] 10 API endpoints
- [x] Data preprocessing
- [x] StandardScaler normalization
- [x] Train/val/test split (60/20/20)
- [x] Model persistence (in-memory)
- [x] Automatic dataset detection
- [x] CORS enabled
- [x] Error handling

---

## Browser Compatibility

### Tested Browsers:
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Edge 90+
- ✅ Safari 14+

### Required Features:
- ES6+ JavaScript support
- Fetch API
- Canvas API (for Chart.js)
- FileReader API
- CSS Grid & Flexbox

---

## Debugging Tips

### Enable Verbose Logging:
```javascript
// In script.js, add at the top:
const DEBUG = true;

// Then in functions:
if (DEBUG) console.log('Training results:', results);
```

### Check Backend Logs:
```bash
# Run backend with verbose output:
uvicorn app:app --reload --log-level debug
```

### Inspect Network Requests:
1. Open browser DevTools (F12)
2. Go to Network tab
3. Filter by XHR/Fetch
4. Check request/response payloads

---

## Success Criteria

All tests pass if:
1. ✅ Dataset uploads without errors
2. ✅ Training completes with all metrics displayed
3. ✅ Individual GA/PSO/DE results show distinct values
4. ✅ Confusion matrix renders with 4 labeled cells
5. ✅ Convergence charts display and toggle correctly
6. ✅ Single predictions return valid results
7. ✅ Batch predictions process CSV without 422 errors
8. ✅ No console errors or warnings

---

## Next Steps After Testing

1. **Production Deployment**:
   - Add authentication/authorization
   - Use external database for model persistence
   - Implement rate limiting
   - Add logging infrastructure

2. **Feature Enhancements**:
   - Support for custom datasets
   - Model comparison tool
   - Export predictions to CSV
   - Real-time training progress bar

3. **Optimization**:
   - Implement caching for predictions
   - Parallel processing for batch predictions
   - GPU acceleration for neural networks
   - Hyperparameter tuning interface

---

**Last Updated**: 2024
**Version**: 1.0.0
**Status**: All critical issues resolved
