# Disease Prediction System

A comprehensive machine learning-based disease prediction system using **Genetic Algorithm (GA)**, **Particle Swarm Optimization (PSO)**, and **Differential Evolution (DE)** for intelligent feature selection, combined with ensemble learning for accurate predictions.

## 🌟 Features

### Core Functionality
- **Multi-Algorithm Feature Selection**: GA, PSO, and DE optimization algorithms
- **Ensemble Learning**: Voting classifier with Logistic Regression, Random Forest, and SVM
- **Neural Network**: MLPClassifier for additional prediction capability
- **Incremental Learning**: Support for adding new data to existing models
- **Multi-Dataset Support**: Breast Cancer, Heart Disease, and Diabetes datasets
- **Seen & Unseen Data Prediction**: Works with both training data and completely new samples

### Advanced Features
- **Dynamic Fitness Function**: Adaptive weighting based on generation
- **Stability Scoring**: Ensures consistent feature selection across iterations
- **Incremental Evaluation**: Phased model evaluation for robust selection
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

### Visualization
- **Convergence Graphs**: Individual and combined plots for GA/PSO/DE
- **Probability Charts**: Visual representation of prediction confidence
- **Confusion Matrix**: Interactive display of classification results
- **Real-time Metrics**: Live dashboard with performance indicators

## 📁 Project Structure

```
Disease-Prediction-System/
│
├── backend/                    # FastAPI Backend
│   ├── app.py                 # Main API endpoints
│   ├── model_utils.py         # ML algorithms (GA/PSO/DE/Training/Prediction)
│   ├── requirements.txt       # Python dependencies
│   ├── scaler.pkl            # Feature scaler (generated after training)
│   ├── ensemble_mask.pkl     # Selected features mask (generated)
│   └── models/               # Trained models directory
│       ├── lr.pkl           # Logistic Regression
│       ├── svm.pkl          # Support Vector Machine
│       ├── rf.pkl           # Random Forest
│       ├── nn.pkl           # Neural Network
│       └── voting.pkl       # Voting Ensemble
│
├── frontend/                  # Web Interface
│   ├── index.html            # Main HTML structure
│   ├── style.css             # Complete styling
│   └── script.js             # JavaScript logic with Chart.js
│
└── README.md                  # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Modern web browser (Chrome, Firefox, Edge)

### Installation

#### 1. Navigate to Backend Directory

```bash
cd Disease-Prediction-System/backend
```

#### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- FastAPI (Web framework)
- Uvicorn (ASGI server)
- scikit-learn (ML library)
- NumPy, Pandas (Data processing)
- joblib (Model serialization)
- python-multipart (File upload support)

#### 3. Start the Backend Server

```bash
uvicorn app:app --reload
```

Or:

```bash
python app.py
```

The API will be available at: `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

#### 4. Open Frontend

Simply open `frontend/index.html` in your web browser, or use a local server:

```bash
cd ../frontend
python -m http.server 8080
```

Then navigate to: `http://localhost:8080`

## 📊 Usage Guide

### Step 1: Upload Dataset

1. Click "Choose File" in the Upload Dataset section
2. Select your CSV file (Breast Cancer / Heart Disease / Diabetes)
3. Click "Upload Dataset"
4. Wait for preprocessing to complete

**Supported Datasets:**
- **Breast Cancer**: Must have a `diagnosis` column (M/B)
- **Heart Disease**: Must have a `target` column (0/1)
- **Diabetes**: Must have an `Outcome` column (0/1)

### Step 2: Train Model

1. Set the number of **Generations** (default: 20)
   - More generations = better optimization but slower
   - Recommended: 20-30 for quick results, 40-50 for best accuracy
   
2. Set the **Population** size (default: 20)
   - Larger population = more exploration but slower
   - Recommended: 20-30

3. Click "Start Training"

**Training Process:**
- GA, PSO, and DE run in parallel
- Feature selection based on fitness function
- Ensemble mask created via majority voting
- Models trained: LR, RF, SVM, Neural Network
- Comprehensive metrics calculated

**What to Expect:**
- Training time: 1-5 minutes (depends on dataset size and parameters)
- Real-time progress indicator
- Convergence graphs show fitness improvement
- Metrics displayed after completion

### Step 3: View Results

**Performance Metrics:**
- Test Accuracy
- Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix

**Feature Selection Summary:**
- Number of features selected by each algorithm
- Final ensemble feature list
- Feature importance visualization

**Convergence Analysis:**
- Combined view: All three algorithms
- Individual views: GA, PSO, DE separately
- Track fitness improvement over generations

### Step 4: Make Predictions

#### Manual Input (Single Prediction)

1. Switch to "Manual Input" tab
2. Enter values for all selected features
3. Click "Predict"

**Prediction Output:**
- **Main Prediction**: POSITIVE or NEGATIVE
- **Confidence Level**: Percentage with visual bar
- **Individual Model Predictions**: LR, RF, SVM, Neural Network
- **Probability Distribution**: Chart showing class probabilities

#### Batch Prediction (Multiple Samples)

1. Switch to "Upload File" tab
2. Prepare a CSV file with feature values
3. Upload the file
4. Click "Predict Batch"

**Batch Results:**
- Table with all predictions
- Confidence scores for each sample
- Color-coded results (green/red)

## 🔬 Technical Details

### Feature Selection Algorithms

#### Genetic Algorithm (GA)
- **Population-based** evolutionary optimization
- **Crossover & Mutation** for exploration
- **Fitness-based selection** ensures quality
- Balances accuracy with feature reduction

#### Particle Swarm Optimization (PSO)
- **Swarm intelligence** inspired by bird flocking
- **Velocity & Position** updates
- **Personal & Global best** tracking
- Fast convergence on optimal solutions

#### Differential Evolution (DE)
- **Mutation, Crossover, Selection** operators
- **Continuous optimization** adapted for binary
- **Robust performance** on complex landscapes
- Excellent for high-dimensional problems

### Fitness Function

```python
fitness = acc_weight * accuracy - alpha * feature_penalty + beta * stability
```

Where:
- `acc_weight`: Dynamic weight increasing over generations
- `alpha`: Feature penalty coefficient (0.02)
- `beta`: Stability bonus coefficient (0.08)

### Ensemble Strategy

**Majority Voting**:
```python
ensemble_mask[i] = 1 if (ga[i] + pso[i] + de[i]) >= 2 else 0
```

**Voting Classifier**:
- Soft voting on probabilities
- Equal weights for LR, RF, SVM
- Final prediction based on average

## 🎯 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/status` | Get training status |
| POST | `/upload` | Upload dataset |
| POST | `/train` | Train models |
| POST | `/predict` | Single prediction |
| POST | `/predict_batch` | Batch predictions |
| GET | `/features` | Get all features |
| GET | `/selected_features` | Get selected features |
| GET | `/metrics` | Get training metrics |

### Request/Response Examples

#### Upload Dataset
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@diabetes.csv"
```

#### Train Model
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"generations": 20, "population": 20}'
```

#### Make Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature1": 1.5, "feature2": 2.3, ...}}'
```

## 🧪 Testing

### Sample Datasets

You can test with these publicly available datasets:

1. **Breast Cancer Wisconsin**: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
2. **Heart Disease**: https://archive.ics.uci.edu/ml/datasets/Heart+Disease
3. **Pima Indians Diabetes**: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

### Expected Accuracy

| Dataset | Expected Accuracy | Feature Reduction |
|---------|------------------|-------------------|
| Breast Cancer | 94-97% | 60-70% |
| Heart Disease | 82-87% | 50-60% |
| Diabetes | 75-80% | 40-50% |

## 📈 Performance Optimization

### Hyperparameter Tuning

**For Better Accuracy:**
- Increase generations: 30-50
- Increase population: 30-40
- Adjust mutation rate in `model_utils.py`

**For Faster Training:**
- Decrease generations: 10-15
- Decrease population: 15-20
- Use fewer models in ensemble

### Model Persistence

Models are automatically saved after training:
- `scaler.pkl`: Feature scaler
- `ensemble_mask.pkl`: Selected features
- `models/*.pkl`: Individual model files

To load saved models:
```python
POST /load_models
```

## 🛠️ Troubleshooting

### Common Issues

**1. Server won't start**
```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Use different port
uvicorn app:app --port 8080
```

**2. CORS errors in browser**
- Backend already has CORS enabled
- Check if API_URL in script.js matches your backend URL

**3. Prediction fails**
- Ensure model is trained first
- Check that all selected features have values
- Verify feature names match exactly

**4. Training takes too long**
- Reduce generations and population
- Use smaller dataset for testing
- Check CPU usage

## 🔮 Future Enhancements

- [ ] Support for multi-class classification
- [ ] Real-time training progress visualization
- [ ] Model comparison and A/B testing
- [ ] Export predictions to CSV
- [ ] Advanced hyperparameter tuning interface
- [ ] Docker containerization
- [ ] Cloud deployment (Azure/AWS)
- [ ] REST API authentication
- [ ] Model versioning and rollback

## 📝 Code Structure

### Backend (`model_utils.py`)

```python
# Feature Selection Algorithms
run_ga()           # Genetic Algorithm
run_pso()          # Particle Swarm Optimization  
run_de()           # Differential Evolution

# Fitness & Evaluation
fitness_function()
incremental_eval()
stability_score()

# Data Processing
preprocess_dataset()
add_new_data()

# Model Training & Prediction
train_models()
make_prediction()
```

### Frontend (`script.js`)

```javascript
// Event Handlers
handleUpload()
handleTrain()
handlePrediction()
handleBatchPrediction()

// Display Functions
displayMetrics()
displaySelectedFeatures()
displayConvergenceCharts()
displayPredictionResult()

// Chart Management
showConvergenceChart()
displayProbabilityChart()
```

## 📄 License

This project is open-source and available for educational and research purposes.

## 👨‍💻 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues, questions, or suggestions, please open an issue on the project repository.

---

**Built with ❤️ using FastAPI, scikit-learn, and Chart.js**
