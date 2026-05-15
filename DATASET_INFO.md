# Dataset Information and Guidelines

## Supported Datasets

### 1. Breast Cancer Wisconsin (Diagnostic)

**Target Column**: `diagnosis`
- **Values**: M (Malignant), B (Benign)
- **Features**: 30 numerical features
- **Expected Accuracy**: 94-97%

**Key Features**:
- radius_mean, texture_mean, perimeter_mean
- area_mean, smoothness_mean, compactness_mean
- concavity_mean, concave points_mean
- symmetry_mean, fractal_dimension_mean
- (plus 20 more features for SE and worst values)

**Download**: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

**Sample CSV Structure**:
```csv
diagnosis,radius_mean,texture_mean,perimeter_mean,...
M,17.99,10.38,122.8,...
B,13.54,14.36,87.46,...
```

---

### 2. Heart Disease

**Target Column**: `target`
- **Values**: 0 (No disease), 1 (Disease present)
- **Features**: 13 numerical features
- **Expected Accuracy**: 82-87%

**Key Features**:
- age, sex, cp (chest pain type)
- trestbps (resting blood pressure)
- chol (serum cholesterol)
- fbs (fasting blood sugar)
- restecg (resting ECG results)
- thalach (max heart rate)
- exang (exercise induced angina)
- oldpeak, slope, ca, thal

**Download**: https://archive.ics.uci.edu/ml/datasets/Heart+Disease

**Sample CSV Structure**:
```csv
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
63,1,3,145,233,1,0,150,0,2.3,0,0,1,1
37,1,2,130,250,0,1,187,0,3.5,0,0,2,1
```

---

### 3. Pima Indians Diabetes

**Target Column**: `Outcome`
- **Values**: 0 (No diabetes), 1 (Diabetes)
- **Features**: 8 numerical features
- **Expected Accuracy**: 75-80%

**Key Features**:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI (Body Mass Index)
- DiabetesPedigreeFunction
- Age

**Download**: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

**Sample CSV Structure**:
```csv
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
```

**Special Note**: This dataset has special zero-handling preprocessing for physiologically impossible zero values.

---

## Dataset Requirements

### File Format
- **Format**: CSV (Comma-Separated Values)
- **Encoding**: UTF-8
- **Size**: Any size (larger datasets take longer to train)

### Required Columns
Your dataset MUST include:
1. **Feature columns**: Numerical values only
2. **Target column**: One of the following:
   - `diagnosis` (for Breast Cancer)
   - `target` (for Heart Disease)
   - `Outcome` (for Diabetes)

### Data Quality
- **Missing Values**: Will be automatically filled with mean values
- **Categorical Features**: Will be automatically encoded
- **Outliers**: Handled by StandardScaler normalization
- **Zeros**: Special handling for Diabetes dataset

---

## Preparing Your Own Dataset

If you want to use a custom dataset:

1. **Format your CSV** with numerical features
2. **Add a target column** named one of:
   - `diagnosis` → for binary classification (M/B or 0/1)
   - `target` → for binary classification (0/1)
   - `Outcome` → for binary classification (0/1)

3. **Ensure binary classification**: Only 0/1 or similar binary values

4. **Example Custom Dataset**:
```csv
feature1,feature2,feature3,feature4,target
1.5,2.3,4.1,0.9,1
2.1,3.2,1.5,1.2,0
0.8,1.9,3.3,0.5,1
```

---

## Sample Data Generation

If you don't have a dataset, you can create a synthetic one:

```python
import pandas as pd
import numpy as np

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
n_features = 10

data = {
    **{f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)},
    'target': np.random.randint(0, 2, n_samples)
}

df = pd.DataFrame(data)
df.to_csv('synthetic_dataset.csv', index=False)
print("Synthetic dataset created: synthetic_dataset.csv")
```

---

## Dataset Split Ratios

The system automatically splits your dataset:

- **Training Set**: 60% (used for model training)
- **Validation Set**: 20% (used for hyperparameter tuning)
- **Test Set**: 20% (used for final evaluation)

All splits are stratified to maintain class balance.

---

## Best Practices

### Data Collection
✅ Ensure data quality and completeness
✅ Remove obvious outliers manually if needed
✅ Verify target column has correct binary values
✅ Check for data imbalance (aim for 40-60% split)

### Feature Engineering
✅ Include domain-relevant features
✅ Avoid redundant or duplicate features
✅ Consider feature interactions if needed
✅ Normalize features manually if extreme scales exist

### Dataset Size
- **Minimum**: 100 samples (may underfit)
- **Recommended**: 500-1000 samples
- **Optimal**: 1000+ samples
- **Large**: 10,000+ samples (slower training)

---

## Troubleshooting

### "Unknown dataset structure" Error
→ Ensure your CSV has one of: `diagnosis`, `target`, or `Outcome` columns

### Low Accuracy
→ Try more generations/population in training
→ Check data quality and feature relevance
→ Ensure sufficient samples per class

### Slow Training
→ Reduce dataset size for testing
→ Decrease generations and population
→ Check for extremely large feature sets

---

## Additional Resources

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [OpenML](https://www.openml.org/)

---

**Need Help?** Check the main README.md or open an issue on the project repository.
