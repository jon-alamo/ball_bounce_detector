import optuna
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from bounce_detector.data_load import load_dataset_from_directory
from bounce_detector.pipeline import run_pipeline_full

# Configuration
DATASET_PATH = "datasets/2022-master-finals-fem"
WINDOW_SIZE = 5
TOLERANCE = 3

# Best "High Recall" Parameters for the heuristic pipeline
BEST_PARAMS = {
    'xw': 2, 'yw': 3,
    'xvw': 3, 'yvw': 2,
    'avw': 4,
    'xaw': 1, 'yaw': 2,
    'mvw': 3,
    'aaw': 4, 'maw': 4,
    'mf': 0.7416,
    'dfx': 0.7651,
    'dfy': 0.892,
    'dfa': 0.8596,
    "dfm": 0.8369,
    'xat': 7.6377,
    'yat': 7.3011,
    'aat': 84.2166,
    'mat': 39.3726,
    'sha': -2,
    'mew': 3,
    'sb': 0.516
}

def extract_features(df, candidate_frame, window=WINDOW_SIZE):
    start = candidate_frame - window
    end = candidate_frame + window
    if start < 0 or end >= len(df):
        return None
    cols = ['vel_x', 'vel_y', 'acc_x', 'acc_y', 'vel_angle', 'vel_module', 'acc_angle', 'acc_module']
    existing_cols = [c for c in cols if c in df.columns]
    
    if not existing_cols:
        return None
        
    subset = df.iloc[start:end+1][existing_cols]
    return subset.values.flatten()

def prepare_dataset():
    print("Loading dataset...")
    # Revert to sample dataset which has labels
    df = load_dataset_from_directory(DATASET_PATH, use_sample=True)
    
    print("Running pipeline...")
    processed_df = run_pipeline_full(df, **BEST_PARAMS)
    
    print("Extracting candidates...")
    candidates = processed_df[processed_df['is_bounce_detected'] == 1].copy()
    
    X_list = []
    y_list = []
    
    # Ground truth frames
    real_bounce_frames = set(df[df['is_bounce'] == True].index)
    
    for frame in candidates.index:
        feat = extract_features(processed_df, frame, WINDOW_SIZE)
        if feat is not None:
            # Check if this candidate is a True Positive
            is_tp = False
            for real_f in real_bounce_frames:
                if abs(real_f - frame) <= TOLERANCE:
                    is_tp = True
                    break
            
            X_list.append(feat)
            y_list.append(1 if is_tp else 0)
            
    return np.array(X_list), np.array(y_list)

def objective(trial):
    X, y = DATASET
    
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 12),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
    }
    
    clf = RandomForestClassifier(**param, random_state=42, n_jobs=-1)
    
    # Stratified K-Fold 
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Optimize for Recall (to minimize False Negatives)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='recall')
    return scores.mean()

if __name__ == "__main__":
    DATASET = prepare_dataset()
    X, y = DATASET
    print(f"Dataset prepared: {len(X)} samples. Positive samples: {sum(y)}")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Recall: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
