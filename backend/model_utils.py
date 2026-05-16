# GA/PSO/DE + prediction logic
import pickle
import numpy as np
import pandas as pd
import random
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import time

warnings.filterwarnings("ignore")

# Global variables to store trained models and data
MODELS = {}
SCALER = None
ENSEMBLE_MASK = None
FEATURE_NAMES = []
X_train = None
y_train = None
X_val = None
y_val = None
X_test = None
y_test = None
DATASET_NAME = None
TARGET = None
TRAINING_RESULTS = None
FITNESS_CACHE = {}


# ============================================
# Utility Functions
# ============================================

def dynamic_weights(gen, total_gens, acc_start=0.6, acc_end=0.9):
    """Calculate dynamic weights for fitness function"""
    frac = gen / max(1, (total_gens - 1))
    acc_w = acc_start + frac * (acc_end - acc_start)
    acc_w = min(max(acc_w, 0.0), 1.0)
    return acc_w


def stability_score(cur_mask, prev_best_mask):
    """Calculate stability score between current and previous mask"""
    if prev_best_mask is None:
        return 0.0
    inter = np.sum((cur_mask == 1) & (prev_best_mask == 1))
    union = np.sum((cur_mask == 1) | (prev_best_mask == 1))
    return (inter / union) if union > 0 else 0.0


def incremental_eval(selected_idx, model_cls=LogisticRegression, phases=1):
    """Incremental evaluation across multiple phases"""
    global X_train, y_train, X_val, y_val
    
    if len(selected_idx) == 0:
        return 0.0
    
    Xsel = X_train[:, selected_idx]
    n = Xsel.shape[0]
    accs = []
    
    for p in range(1, phases + 1):
        upto = int(n * (p / phases))
        if upto < 2:
            upto = min(2, n)
        
        clf = model_cls(max_iter=500)
        clf.fit(Xsel[:upto], y_train[:upto])
        preds = clf.predict(X_val[:, selected_idx])
        accs.append(accuracy_score(y_val, preds))
    
    return np.mean(accs)


def fitness_function(mask, gen, total_gens, prev_best_mask=None, alpha=0.02, beta=0.08):
    """
    Fitness function with dynamic weights, stability, and incremental evaluation
    """
    global X_train, y_train, X_val, y_val
    
    if mask.sum() == 0:
        return -999.0

    selected_idx = np.where(mask == 1)[0]
    cache_key = tuple(selected_idx.tolist())
    
    try:
        if cache_key not in FITNESS_CACHE:
            FITNESS_CACHE[cache_key] = incremental_eval(
                selected_idx,
                model_cls=LogisticRegression,
                phases=1
            )
        acc = FITNESS_CACHE[cache_key]
    except Exception as e:
        clf = LogisticRegression(max_iter=500)
        clf.fit(X_train[:, selected_idx], y_train)
        acc = accuracy_score(y_val, clf.predict(X_val[:, selected_idx]))

    acc_w = dynamic_weights(gen, total_gens)
    feat_pen = (mask.sum() / len(mask))
    stab = stability_score(mask, prev_best_mask)

    fitness = acc_w * acc - alpha * feat_pen + beta * stab
    return float(fitness)


def random_mask(n, p=0.25):
    """Generate random binary mask"""
    return np.array([1 if random.random() < p else 0 for _ in range(n)], dtype=int)


# ============================================
# Genetic Algorithm
# ============================================

def run_ga(gens=25, pop=24, mutation_rate=0.12):
    """Run Genetic Algorithm for feature selection"""
    global X_train
    
    n = X_train.shape[1]
    population = [random_mask(n, p=0.25) for _ in range(pop)]
    best_mask = None
    best_fit = -1e9
    history = []
    prev_best_mask = None

    for g in range(gens):
        fits = []
        for m in population:
            f = fitness_function(m, g, gens, prev_best_mask)
            fits.append(f)
        fits = np.array(fits)

        idx_best = np.argmax(fits)
        if fits[idx_best] > best_fit:
            best_fit = fits[idx_best]
            best_mask = population[idx_best].copy()

        prev_best_mask = best_mask.copy() if best_mask is not None else None
        history.append(best_fit)

        sorted_idx = fits.argsort()[::-1]
        survivors = [population[i] for i in sorted_idx[: pop // 2]]

        new_pop = survivors.copy()
        while len(new_pop) < pop:
            p1, p2 = random.sample(survivors, 2)
            pt = random.randint(1, n - 1)
            c1 = np.concatenate((p1[:pt], p2[pt:])).copy()
            c2 = np.concatenate((p2[:pt], p1[pt:])).copy()
            for c in [c1, c2]:
                for i in range(n):
                    if random.random() < mutation_rate:
                        c[i] = 1 - c[i]
                if len(new_pop) < pop:
                    new_pop.append(c)
        population = new_pop

    return best_mask.astype(int), history


# ============================================
# Particle Swarm Optimization
# ============================================

def run_pso(gens=25, pop=24, w=0.6, c1=1.4, c2=1.4):
    """Run Particle Swarm Optimization for feature selection"""
    global X_train
    
    n = X_train.shape[1]
    particles = [random_mask(n, p=0.25) for _ in range(pop)]
    velocities = [np.random.uniform(-1, 1, n) for _ in range(pop)]
    personal_best = particles.copy()
    personal_best_score = [-1e9] * pop
    global_best = None
    global_best_score = -1e9
    history = []
    prev_best_mask = None

    for g in range(gens):
        for i in range(pop):
            score = fitness_function(particles[i], g, gens, prev_best_mask)
            if score > personal_best_score[i]:
                personal_best_score[i] = score
                personal_best[i] = particles[i].copy()
            if score > global_best_score:
                global_best_score = score
                global_best = particles[i].copy()

        prev_best_mask = global_best.copy() if global_best is not None else None
        history.append(global_best_score)

        for i in range(pop):
            r1 = np.random.rand(n)
            r2 = np.random.rand(n)
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (personal_best[i] - particles[i]) + 
                           c2 * r2 * (global_best - particles[i]))
            probs = 1.0 / (1.0 + np.exp(-velocities[i]))
            particles[i] = np.array([1 if random.random() < p else 0 for p in probs], dtype=int)

    return global_best.astype(int), history


# ============================================
# Differential Evolution
# ============================================

def run_de(gens=25, pop=24, F=0.7, CR=0.6):
    """Run Differential Evolution for feature selection"""
    global X_train
    
    n = X_train.shape[1]
    population = [random_mask(n, p=0.25) for _ in range(pop)]
    best_mask = None
    best_fit = -1e9
    history = []
    prev_best_mask = None

    for g in range(gens):
        fits = [fitness_function(ind, g, gens, prev_best_mask) for ind in population]

        for i in range(pop):
            idxs = list(range(pop))
            idxs.remove(i)
            chosen = np.random.choice(idxs, 3, replace=False)
            a, b, c = [population[j] for j in chosen]

            mutant = np.clip(a + F * (b - c), 0, 1)
            trial = np.array([int(mutant[j]) if random.random() < CR else population[i][j] 
                            for j in range(n)], dtype=int)

            f_trial = fitness_function(trial, g, gens, prev_best_mask)
            if f_trial > fits[i]:
                population[i] = trial
                fits[i] = f_trial

            if fits[i] > best_fit:
                best_fit = fits[i]
                best_mask = population[i].copy()

        prev_best_mask = best_mask.copy() if best_mask is not None else None
        history.append(best_fit)

    return best_mask.astype(int), history


# ============================================
# Data Processing
# ============================================

def preprocess_dataset(df, filename):
    """Preprocess uploaded dataset with multi-dataset support"""
    global SCALER, FEATURE_NAMES, X_train, y_train, X_val, y_val, X_test, y_test
    global DATASET_NAME, TARGET
    
    # Drop CSV artifacts and identifier columns that do not help prediction.
    df = df.drop(columns=[col for col in df.columns if str(col).startswith("Unnamed")], errors="ignore")
    df = df.drop(columns=["id"], errors="ignore")

    # Auto-detect dataset and target column
    if "diagnosis" in df.columns:
        DATASET_NAME = "Breast Cancer"
        TARGET = "diagnosis"
    elif "target" in df.columns:
        DATASET_NAME = "Heart Disease"
        TARGET = "target"
    elif "Outcome" in df.columns:
        DATASET_NAME = "Diabetes"
        TARGET = "Outcome"
    else:
        raise ValueError("Unknown dataset structure. Cannot detect target column.")

    # Special fix for Diabetes dataset
    if DATASET_NAME == "Diabetes":
        zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        for col in zero_columns:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
                df[col] = df[col].fillna(df[col].mean())

    # Split features and target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    FEATURE_NAMES = X.columns.tolist()

    # Convert to numeric and handle NaN
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean(numeric_only=True))
    X = X.fillna(0)

    # Encode target if categorical
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    # Feature scaling
    SCALER = StandardScaler()
    X_scaled = SCALER.fit_transform(X)

    # Train/Val/Test split (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    return {
        "dataset_name": DATASET_NAME,
        "target": TARGET,
        "features": FEATURE_NAMES,
        "train_shape": X_train.shape,
        "val_shape": X_val.shape,
        "test_shape": X_test.shape
    }


def add_new_data(new_df):
    """Incrementally add new data to training set"""
    global X_train, y_train, SCALER, TARGET, DATASET_NAME

    X_new = new_df.drop(columns=[TARGET])
    y_new = new_df[TARGET]

    X_new = X_new.apply(pd.to_numeric, errors="coerce")
    X_new = X_new.fillna(X_new.mean(numeric_only=True))
    X_new = X_new.fillna(0)

    if y_new.dtype == "object":
        y_new = LabelEncoder().fit_transform(y_new)

    if DATASET_NAME == "Diabetes":
        zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        for col in zero_columns:
            if col in X_new.columns:
                X_new[col] = X_new[col].replace(0, np.nan)
                X_new[col] = X_new[col].fillna(X_new[col].mean())

    X_new_scaled = SCALER.transform(X_new)
    X_train = np.vstack([X_train, X_new_scaled])
    y_train = np.concatenate([y_train, y_new])

    return {"new_samples": X_new.shape[0], "total_samples": X_train.shape[0]}


# ============================================
# Model Training
# ============================================

def evaluate_mask_metrics(mask, mask_name="Mask", optimization_time_sec=None):
    """Evaluate a single mask and return comprehensive metrics"""
    global X_train, y_train, X_val, y_val, X_test, y_test
    
    eval_start = time.time()
    idx = np.where(mask == 1)[0]
    if len(idx) == 0:
        return None
    
    # Train simple LR model on this mask
    clf = LogisticRegression(max_iter=3000)
    clf.fit(X_train[:, idx], y_train)
    
    # Get predictions
    val_preds = clf.predict(X_val[:, idx])
    test_preds = clf.predict(X_test[:, idx])
    
    # Calculate metrics
    metrics = {
        'algorithm': mask_name,
        'val_accuracy': float(accuracy_score(y_val, val_preds)),
        'test_accuracy': float(accuracy_score(y_test, test_preds)),
        'precision': float(precision_score(y_test, test_preds, zero_division=0)),
        'recall': float(recall_score(y_test, test_preds, zero_division=0)),
        'f1_score': float(f1_score(y_test, test_preds, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, test_preds).tolist(),
        'selected_features': int(mask.sum()),
        'feature_indexes': [int(i) for i in idx.tolist()]
    }
    
    # Add ROC-AUC if possible
    try:
        proba = clf.predict_proba(X_test[:, idx])[:, 1]
        metrics['roc_auc'] = float(roc_auc_score(y_test, proba))
    except:
        metrics['roc_auc'] = None

    metrics['evaluation_time_sec'] = round(time.time() - eval_start, 4)
    if optimization_time_sec is not None:
        metrics['optimization_time_sec'] = round(float(optimization_time_sec), 4)
        metrics['time_taken_sec'] = round(float(optimization_time_sec), 4)
    else:
        metrics['time_taken_sec'] = metrics['evaluation_time_sec']
    
    return metrics


def train_models(gens=20, pop=20):
    """Run optimization and train ensemble models"""
    global ENSEMBLE_MASK, MODELS, X_train, y_train, X_val, y_val, X_test, y_test
    global FITNESS_CACHE
    
    FITNESS_CACHE = {}
    training_start = time.time()

    # Run optimizers
    ga_start = time.time()
    ga_mask, ga_hist = run_ga(gens=gens, pop=pop, mutation_rate=0.12)
    ga_time_sec = time.time() - ga_start

    pso_start = time.time()
    pso_mask, pso_hist = run_pso(gens=gens, pop=pop)
    pso_time_sec = time.time() - pso_start

    de_start = time.time()
    de_mask, de_hist = run_de(gens=gens, pop=pop)
    de_time_sec = time.time() - de_start
    
    # Evaluate individual masks
    ga_metrics = evaluate_mask_metrics(ga_mask, "GA", ga_time_sec)
    pso_metrics = evaluate_mask_metrics(pso_mask, "PSO", pso_time_sec)
    de_metrics = evaluate_mask_metrics(de_mask, "DE", de_time_sec)
    
    # Create ensemble mask (majority vote)
    ENSEMBLE_MASK = np.array([
        1 if (ga_mask[i] + pso_mask[i] + de_mask[i]) >= 2 else 0 
        for i in range(len(ga_mask))
    ], dtype=int)
    
    # Get selected feature indices
    sel_idx = np.where(ENSEMBLE_MASK == 1)[0]
    
    if len(sel_idx) == 0:
        raise RuntimeError("Ensemble mask selected zero features")
    
    # Train individual models
    lr_model = LogisticRegression(max_iter=2000, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42)
    svm_model = SVC(probability=True, kernel='rbf', gamma='scale', random_state=42)
    nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', 
                            max_iter=800, random_state=42)
    
    # Train voting ensemble
    voting = VotingClassifier(
        estimators=[('lr', lr_model), ('rf', rf_model), ('svc', svm_model)],
        voting='soft',
        n_jobs=1
    )
    voting.fit(X_train[:, sel_idx], y_train)
    
    # Train neural network separately
    nn_model.fit(X_train[:, sel_idx], y_train)
    
    # Store models - get trained models from voting classifier
    MODELS = {
        'voting': voting,
        'lr': voting.named_estimators_['lr'],
        'rf': voting.named_estimators_['rf'],
        'svm': voting.named_estimators_['svc'],
        'nn': nn_model,
        'ga_mask': ga_mask,
        'pso_mask': pso_mask,
        'de_mask': de_mask,
        'ensemble_mask': ENSEMBLE_MASK
    }
    
    # Calculate ensemble metrics
    val_preds = voting.predict(X_val[:, sel_idx])
    test_preds = voting.predict(X_test[:, sel_idx])
    
    ensemble_metrics = {
        'algorithm': 'Ensemble',
        'val_accuracy': float(accuracy_score(y_val, val_preds)),
        'test_accuracy': float(accuracy_score(y_test, test_preds)),
        'precision': float(precision_score(y_test, test_preds, zero_division=0)),
        'recall': float(recall_score(y_test, test_preds, zero_division=0)),
        'f1_score': float(f1_score(y_test, test_preds, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, test_preds).tolist(),
        'selected_features_count': int(ENSEMBLE_MASK.sum()),
        'feature_indexes': [int(i) for i in sel_idx.tolist()],
        'ga_features': int(ga_mask.sum()),
        'pso_features': int(pso_mask.sum()),
        'de_features': int(de_mask.sum())
    }
    
    # Add ROC-AUC if binary classification
    try:
        proba = voting.predict_proba(X_test[:, sel_idx])[:, 1]
        ensemble_metrics['roc_auc'] = float(roc_auc_score(y_test, proba))
    except:
        ensemble_metrics['roc_auc'] = None

    ensemble_metrics['time_taken_sec'] = round(time.time() - training_start, 4)
    
    results = {
        'metrics': ensemble_metrics,
        'ga_metrics': ga_metrics,
        'pso_metrics': pso_metrics,
        'de_metrics': de_metrics,
        'convergence': {
            'ga': [float(x) for x in ga_hist],
            'pso': [float(x) for x in pso_hist],
            'de': [float(x) for x in de_hist]
        },
        'selected_features': get_selected_features()
    }
    
    # Store results in global variable for later retrieval
    global TRAINING_RESULTS
    TRAINING_RESULTS = results
    
    return results


def get_selected_features():
    """Get list of selected feature names"""
    global ENSEMBLE_MASK, FEATURE_NAMES
    
    if ENSEMBLE_MASK is None:
        return []
    
    return [FEATURE_NAMES[i] for i in range(len(ENSEMBLE_MASK)) if ENSEMBLE_MASK[i] == 1]


# ============================================
# Prediction
# ============================================

def make_prediction(input_data):
    """Make prediction on new data (unseen)"""
    global MODELS, SCALER, ENSEMBLE_MASK, FEATURE_NAMES
    
    if MODELS is None or 'voting' not in MODELS:
        raise ValueError("Models not trained yet")
    
    # Convert input to DataFrame
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = pd.DataFrame(input_data)
    
    # Ensure all features are present
    for feat in FEATURE_NAMES:
        if feat not in df.columns:
            df[feat] = 0
    
    # Reorder columns to match training data
    df = df[FEATURE_NAMES]
    
    # Preprocess
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0)
    
    # Scale
    X_scaled = SCALER.transform(df)
    
    # Select features
    sel_idx = np.where(ENSEMBLE_MASK == 1)[0]
    X_selected = X_scaled[:, sel_idx]
    
    # Predict with all models
    voting_pred = MODELS['voting'].predict(X_selected)
    voting_proba = MODELS['voting'].predict_proba(X_selected)
    
    lr_pred = MODELS['lr'].predict(X_selected)
    lr_proba = MODELS['lr'].predict_proba(X_selected)
    
    rf_pred = MODELS['rf'].predict(X_selected)
    rf_proba = MODELS['rf'].predict_proba(X_selected)
    
    svm_pred = MODELS['svm'].predict(X_selected)
    svm_proba = MODELS['svm'].predict_proba(X_selected)
    
    nn_pred = MODELS['nn'].predict(X_selected)
    nn_proba = MODELS['nn'].predict_proba(X_selected)
    
    results = {
        'prediction': int(voting_pred[0]),
        'probability': float(voting_proba[0][1]),
        'confidence': float(max(voting_proba[0])),
        'voting_proba': voting_proba[0].tolist(),
        'nn_prediction': int(nn_pred[0]),
        'nn_proba': nn_proba[0].tolist(),
        'lr_proba': lr_proba[0].tolist(),
        'rf_proba': rf_proba[0].tolist(),
        'svm_proba': svm_proba[0].tolist(),
        'individual_predictions': {
            'lr': int(lr_pred[0]),
            'rf': int(rf_pred[0]),
            'svm': int(svm_pred[0]),
            'nn': int(nn_pred[0])
        }
    }
    
    return results


def save_models():
    """Save trained models and training results to disk"""
    global MODELS, SCALER, ENSEMBLE_MASK, TRAINING_RESULTS
    
    import joblib
    
    joblib.dump(SCALER, 'scaler.pkl')
    joblib.dump(ENSEMBLE_MASK, 'ensemble_mask.pkl')
    joblib.dump(MODELS['lr'], 'models/lr.pkl')
    joblib.dump(MODELS['rf'], 'models/rf.pkl')
    joblib.dump(MODELS['svm'], 'models/svm.pkl')
    joblib.dump(MODELS['nn'], 'models/nn.pkl')
    joblib.dump(MODELS['voting'], 'models/voting.pkl')
    
    # Save training results if available
    if TRAINING_RESULTS:
        joblib.dump(TRAINING_RESULTS, 'training_results.pkl')


def load_saved_models():
    """Load models and training results from disk"""
    global MODELS, SCALER, ENSEMBLE_MASK, TRAINING_RESULTS
    
    import joblib
    import os
    
    SCALER = joblib.load('scaler.pkl')
    ENSEMBLE_MASK = joblib.load('ensemble_mask.pkl')
    
    MODELS = {
        'lr': joblib.load('models/lr.pkl'),
        'rf': joblib.load('models/rf.pkl'),
        'svm': joblib.load('models/svm.pkl'),
        'nn': joblib.load('models/nn.pkl'),
        'voting': joblib.load('models/voting.pkl'),
        'ensemble_mask': ENSEMBLE_MASK
    }
    
    # Load training results if available
    if os.path.exists('training_results.pkl'):
        TRAINING_RESULTS = joblib.load('training_results.pkl')
