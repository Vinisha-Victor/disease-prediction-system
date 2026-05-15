# Quick Start Guide

## 🚀 Getting Started in 3 Minutes

### Step 1: Install Dependencies
```bash
cd Disease-Prediction-System
pip install -r backend/requirements.txt
```

### Step 2: Start Backend
```bash
cd backend
python app.py
```
Server starts at: `http://localhost:8000`

### Step 3: Open Frontend
Double-click: `frontend/index.html`

---

## 📋 Usage Flow

### 1. Upload Dataset
- Click **Browse** → Select dataset (breast_cancer.csv / heart.csv / diabetes.csv)
- Click **Upload Dataset**
- ✅ Dataset info displays

### 2. Train Model
- Set **Generations**: 20 (recommended)
- Set **Population Size**: 30 (recommended)
- Click **Start Training**
- ⏱️ Wait 30-60 seconds

### 3. View Results
After training, you'll see:
- **Individual Algorithm Results**: GA, PSO, DE cards with metrics
- **Ensemble Metrics**: Overall model performance
- **Confusion Matrix**: 2x2 grid with predictions
- **Convergence Charts**: Toggle between Combined/GA/PSO/DE views
- **Selected Features**: List of chosen features

### 4. Make Predictions

#### Single Prediction:
- Enter feature values (comma-separated)
- Click **Predict**
- See prediction, confidence, probabilities

#### Batch Prediction:
- Prepare CSV with same features as training data
- Click **Browse** → Select CSV
- Click **Predict Batch**
- View table with all predictions

---

## 🎯 Key Features

### Individual Algorithm Results ✅
Each algorithm (GA, PSO, DE) shows:
- Number of features selected
- Accuracy percentage
- F1 Score

### Confusion Matrix ✅
Visual 2x2 grid showing:
- **True Negative** (green border)
- **False Positive** (red border)
- **False Negative** (red border)
- **True Positive** (green border)

### Convergence Charts ✅
Interactive charts with:
- Combined view (all 3 algorithms)
- Individual views (GA/PSO/DE)
- Color-coded lines
- Smooth animations

### Batch Prediction ✅
Upload CSV files for bulk predictions:
- Automatic validation
- Clear error messages
- Color-coded results table

---

## 📊 Example Results

### Breast Cancer Dataset:
```
Training Time: ~42 seconds

Individual Results:
├── GA: 12 features, 94.2% accuracy, 0.93 F1
├── PSO: 10 features, 93.8% accuracy, 0.92 F1
└── DE: 11 features, 94.5% accuracy, 0.94 F1

Ensemble: 95.1% accuracy, 0.94 F1, 0.96 ROC AUC

Confusion Matrix:
  55 | 2
  ---+---
  3  | 54
```

---

## 🔧 Troubleshooting

### Issue: 422 Error on Batch Prediction
**Solution**: Ensure CSV headers exactly match training features

### Issue: Graphs Not Showing
**Solution**: 
1. Check browser console for errors
2. Verify Chart.js loaded
3. Clear cache and reload

### Issue: Training Takes Too Long
**Solution**: Reduce generations or population size

---

## 📁 Project Structure

```
Disease-Prediction-System/
│
├── backend/
│   ├── app.py                    # FastAPI server
│   ├── model_utils.py            # ML algorithms (GA/PSO/DE)
│   └── requirements.txt          # Python dependencies
│
├── frontend/
│   ├── index.html                # Main UI
│   ├── script.js                 # API calls & Chart.js
│   └── style.css                 # Styling
│
├── datasets/                     # Sample datasets
├── docs/                         # Documentation
├── test_batch.csv               # Sample test file
├── README.md                    # Full documentation
├── TESTING_GUIDE.md             # Testing instructions
└── ISSUE_RESOLUTION_SUMMARY.md  # Fix details
```

---

## 🔗 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/upload` | POST | Upload dataset |
| `/status` | GET | Check training status |
| `/train` | POST | Train model |
| `/predict` | POST | Single prediction |
| `/predict_batch` | POST | Batch prediction |
| `/features` | GET | Get selected features |
| `/convergence` | GET | Get convergence data |
| `/reset` | POST | Reset system |
| `/health` | GET | Health check |
| `/docs` | GET | API documentation |

---

## 📚 Documentation

- **README.md**: Complete project overview
- **API_DOCUMENTATION.md**: Detailed API reference
- **TESTING_GUIDE.md**: Step-by-step testing instructions
- **ISSUE_RESOLUTION_SUMMARY.md**: Recent fixes and changes
- **DATASET_INFO.md**: Dataset descriptions
- **PROJECT_SUMMARY.md**: Project architecture

---

## 🎓 ML Algorithms

### Genetic Algorithm (GA)
- Binary encoding for feature selection
- Mutation rate: 0.12
- Crossover: Two-point
- Selection: Tournament

### Particle Swarm Optimization (PSO)
- Velocity-based updates
- Inertia weight: 0.6
- Cognitive factor: 1.4
- Social factor: 1.4

### Differential Evolution (DE)
- Mutation factor: 0.7
- Crossover rate: 0.6
- Strategy: rand/1/bin

### Fitness Function
```
fitness = acc_weight * accuracy 
        - feature_penalty * (num_features / total_features)
        + stability_weight * stability_score
```

---

## 💡 Tips

1. **Training Time**: 
   - Fewer generations = faster training
   - More population = better results but slower

2. **Feature Selection**:
   - GA tends to select more features
   - PSO finds balance
   - DE often most aggressive

3. **Ensemble**:
   - Combines Logistic Regression, Random Forest, SVM
   - Usually outperforms individual algorithms

4. **Datasets**:
   - Breast Cancer: 30 features, 569 samples
   - Heart Disease: 13 features, 303 samples
   - Diabetes: 8 features, 768 samples

---

## 🏆 Success Criteria

✅ All features working:
- [x] Dataset upload
- [x] Model training
- [x] Individual algorithm results (GA/PSO/DE)
- [x] Ensemble metrics
- [x] Confusion matrix
- [x] Convergence charts
- [x] Single prediction
- [x] Batch prediction

---

## 🚨 Important Notes

1. **Browser Compatibility**: Use Chrome/Firefox/Edge (latest versions)
2. **File Format**: CSV files must have headers matching training features
3. **Feature Names**: Case-sensitive and exact match required
4. **Missing Values**: Not supported (ensure clean data)

---

## 📞 Need Help?

Check documentation files:
1. **Quick issue?** → This guide
2. **Testing?** → TESTING_GUIDE.md
3. **API?** → API_DOCUMENTATION.md
4. **Recent fixes?** → ISSUE_RESOLUTION_SUMMARY.md
5. **Full details?** → README.md

---

**Version**: 1.0.0  
**Status**: ✅ Production Ready  
**Last Updated**: 2024

---

## 🎉 You're Ready!

Now start the backend, open the frontend, and begin making predictions!

```bash
# Terminal 1 - Start backend
cd backend
python app.py

# Terminal 2 - Or open frontend
# Just double-click frontend/index.html
```

**Happy Predicting! 🎯**
