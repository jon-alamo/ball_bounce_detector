import pandas as pd
import numpy as np
import joblib
import os


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

    # Columns required for the classifier
    # Ensure these match what the model was trained on
    # Correct names matching b_compute_polar.py
    cols = ['vel_x', 'vel_y', 'acc_x', 'acc_y', 'vel_angle', 'vel_module', 'acc_angle', 'acc_module']
    
    # Check if cols exist, fill missing with 0 or handle error
    # For robust pipeline, we assume these computed columns exist.
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        # If polar features are missing (e.g. XY pipeline), we can't use this classifier 
        # unless we re-train specifically for XY features. 
        # For now, we return None to skip or warn.
        return None
    
    subset = df.iloc[start:end+1][cols]
    
    # Flatten: [vx_t-5, ..., vx_t, ..., vx_t+5, vy_t-5, ...]
    feature_vector = subset.values.flatten()
    
    return feature_vector


def classify_candidates(df, model_path=None, window=5):
    """
    Filters detected bounces ('is_bounce_detected' == 1) using a pre-trained ML model.
    Only candidates predicted as '1' (Real Bounce) are kept.
    """
    if model_path is None:
        # Default to assets folder relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level from 'pipeline' to 'bounce_detector' then into 'assets'
        model_path = os.path.join(current_dir, '..', 'assets', 'bounce_classifier.pkl')

    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}. Skipping classification.")
        return df

    df = df.copy()
    
    # Load model (consider caching this if performance is critical or loading once globally)
    try:
        clf = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return df
    
    detected_indices = df.index[df['is_bounce_detected'] == 1].tolist()
    
    if not detected_indices:
        return df
        
    # Prepare batch features
    X = []
    valid_indices = []
    
    for idx in detected_indices:
        features = extract_features(df, idx, window=window)
        if features is not None:
            X.append(features)
            valid_indices.append(idx)
        else:
            # If we can't extract features (e.g. edge of video), 
            # we might choose to keep it (conservative) or drop it.
            # Here we keep it to avoid deleting valid start/end bounces.
            pass
            
    if not X:
        return df
        
    X = np.array(X)
    
    # Verify shape matches model expectation
    # (Optional robust check could go here)
    
    # Predict
    preds = clf.predict(X)
    
    # Apply results
    for i, idx in enumerate(valid_indices):
        if preds[i] == 0:  # Classified as False Positive (Ghost)
            df.at[idx, 'is_bounce_detected'] = 0
            
    return df
