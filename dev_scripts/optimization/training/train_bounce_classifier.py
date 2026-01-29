import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import joblib

from bounce_detector.data_load import load_dataset_from_directory
from bounce_detector.pipeline import run_pipeline_full
from bounce_detector.compute_results import compare_with_sample

# --- Configuration ---
DATASET_PATH = "datasets/2022-master-finals-fem"
WINDOW_SIZE = 5  # Frames before and after to extract features
TOLERANCE = 3    # Tolerance for labeling candidates (matching ground truth)

# Best "High Recall" Parameters (from previous optimization)
BEST_PARAMS = {
    "xw": 2,
    "yw": 3,
    "xvw": 3,
    "yvw": 2,
    "avw": 4,
    "xaw": 1,
    "yaw": 2,
    "mvw": 3,
    "aaw": 4,
    "maw": 4,
    "mf": 0.7416,
    "dfx": 0.7651,
    "dfy": 0.892,
    "dfa": 0.8596,
    "dfm": 0.8369,
    "xat": 7.6377,
    "yat": 7.3011,
    "aat": 84.2166,
    "mat": 39.3726,
    "sha": -2,
    "mew": 3,
    "sb": 0.516
}

def extract_features(df, candidate_frame, window=5):
    """
    Extracts a feature vector for a given candidate frame including neighbors.
    Features: vx, vy, ax, ay, ang_v, mod_v, ang_a, mod_a over the window.
    """
    start = candidate_frame - window
    end = candidate_frame + window
    
    # Check boundaries
    if start < 0 or end >= len(df):
        return None

    # Columns of interest from the pipeline computation
    # Note: These columns must exist in the df passed (processed_df)
    # We expect the pipeline to have added: 'vel_x', 'vel_y', 'acc_x', 'acc_y', etc.
    # If standard pipeline names differ, we check src/pipeline implementations.
    # Based on a_compute_xy.py usually we have: 'vel_x', 'vel_y', 'acc_x', 'acc_y'
    # Based on b_compute_polar.py: 'vel_angle', 'vel_module', 'acc_angle', 'acc_module'
    
    cols = ['vel_x', 'vel_y', 'acc_x', 'acc_y', 'vel_angle', 'vel_module', 'acc_angle', 'acc_module']
    
    # Check if cols exist, otherwise handle graceful failure or zeros
    existing_cols = [c for c in cols if c in df.columns]
    
    subset = df.iloc[start:end+1][existing_cols]
    
    # Flatten the features: [vx_t-5, ..., vx_t, ..., vx_t+5, vy_t-5, ...]
    feature_vector = subset.values.flatten()
    
    return feature_vector

def main():
    print("Loading dataset...")
    df = load_dataset_from_directory(DATASET_PATH, use_sample=True)
    
    print("Running Stage 1: Candidate Generation (High Recall Pipeline)...")
    # We use run_pipeline_full because BEST_PARAMS includes polar params
    processed_df = run_pipeline_full(df, **BEST_PARAMS)
    
    # Label candidates
    # compare_with_sample adds 'true_positive', 'false_positive' columns
    labeled_df = compare_with_sample(processed_df, tolerance=TOLERANCE)
    
    # Extract candidates
    # A candidate is any frame where 'is_bounce_detected' == 1
    candidate_indices = labeled_df.index[labeled_df['is_bounce_detected'] == 1].tolist()
    
    X = []
    y = []
    valid_indices = []
    
    print(f"Extracting features for {len(candidate_indices)} candidates...")
    
    for idx in candidate_indices:
        features = extract_features(labeled_df, idx, window=WINDOW_SIZE)
        if features is not None:
            # Determine label: 1 if it was a TP, 0 if FP
            # compare_with_sample sets 'true_positive' = 1 for hits
            label = 1 if labeled_df.at[idx, 'true_positive'] == 1 else 0
            
            X.append(features)
            y.append(label)
            valid_indices.append(idx)
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dataset assembled. Total Samples: {len(y)}")
    print(f"Positives (TP): {np.sum(y == 1)}")
    print(f"Negatives (FP): {np.sum(y == 0)}")
    
    if len(y) < 10:
        print("Not enough data to train a classifier.")
        return

    # --- Train and Evaluate Stage 2 Classifier ---
    print("\nTraining Random Forest Classifier (CV)...")
    
    # Updated with optimized parameters (from optimize_bounce_classifier.py)
    clf = RandomForestClassifier(
        n_estimators=184, 
        max_depth=19, 
        min_samples_split=4, 
        min_samples_leaf=4, 
        bootstrap=True, 
        criterion='entropy',
        random_state=42
    )
    
    # Cross-validated predictions
    y_pred = cross_val_predict(clf, X, y, cv=5)
    
    # Statistics
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    print("-" * 30)
    print("STAGE 2 CLASSIFIER PERFORMANCE (Cross-Validation)")
    print("-" * 30)
    print(f"Precision: {prec:.2%} (Percentage of survivors that are real)")
    print(f"Recall:    {rec:.2%} (Percentage of real candidates kept)")
    print(f"F1 Score:  {f1:.2%}")
    print("-" * 30)
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    
    # --- Final Pipeline Simulation ---
    # Total Real Bounces in Dataset
    total_real = labeled_df['is_bounce'].sum()
    
    # Stage 1 Performance
    tp_s1 = np.sum(y == 1)
    fp_s1 = np.sum(y == 0)
    fn_s1 = total_real - tp_s1 # Approximate, assuming candidates matched all possible TPs
                                # (Wait, compare_with_sample logic matches unique so this holds)
    
    # Stage 2 Filtered
    # Survivors: Predicted as 1 (True)
    # New TP = Real bounces (y=1) that were predicted as 1
    new_tp = np.sum((y == 1) & (y_pred == 1))
    
    # New FP = Ghosts (y=0) that were predicted as 1
    new_fp = np.sum((y == 0) & (y_pred == 1))
    
    # New FN = Old FN + Real candidates rejected by S2
    rejected_tp = np.sum((y == 1) & (y_pred == 0))
    new_fn = fn_s1 + rejected_tp
    
    final_precision = new_tp / (new_tp + new_fp) if (new_tp + new_fp) > 0 else 0
    final_recall = new_tp / total_real if total_real > 0 else 0
    final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0

    print("\n" + "=" * 30)
    print("FINAL TWO-STAGE PIPELINE RESULTS")
    print("=" * 30)
    print(f"Old Precision (Stage 1): {tp_s1 / (tp_s1 + fp_s1):.2%}")
    print(f"Old Recall (Stage 1):    {tp_s1 / total_real:.2%}")
    print("-" * 30)
    print(f"New Precision: {final_precision:.2%}")
    print(f"New Recall:    {final_recall:.2%}")
    print(f"New F1 Score:  {final_f1:.2%}")
    print(f"True Positives: {new_tp} (Merged/Filtered)")
    print(f"False Positives: {new_fp}")
    print(f"False Negatives: {new_fn}")
    print("=" * 30)

    # Train final model on all data
    clf.fit(X, y)
    
    # Feature Importance
    importances = clf.feature_importances_
    # Mapping back feature names is a bit complex due to flattening, 
    # but we can print the top raw index.
    print("\nFeature shape per sample:", inputs_shape := X.shape[1])
    # print("Top 10 feature indices:", np.argsort(importances)[::-1][:10])

    print("\nSaving model to bounce_classifier.pkl...")
    joblib.dump(clf, "bounce_classifier.pkl")
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
