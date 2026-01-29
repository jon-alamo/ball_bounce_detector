import pandas as pd
import numpy as np
import joblib
import os


def extract_features(df, candidate_frame, window=5):
    """
    Extracts a feature vector for a candidate frame from its temporal window.
    Features: Flattened array of velocity and acceleration (XY and Polar) over [t-w, t+w].
    """
    start = candidate_frame - window
    end = candidate_frame + window
    
    if start < 0 or end >= len(df):
        return None

    # Required feature columns (order matters for the trained model)
    cols = ['vel_x', 'vel_y', 'acc_x', 'acc_y', 
            'vel_angle', 'vel_module', 'acc_angle', 'acc_module']
    
    # Verify existence
    if not all(c in df.columns for c in cols):
        return None
    
    subset = df.iloc[start:end+1][cols]
    
    # Flatten -> [v_x_0, v_y_0, ..., v_x_N, v_y_N, ...] if flattened by row, 
    # but pandas values.flatten() is row-major by default (C-style).
    # Ensure this matches training logic.
    return subset.values.flatten()


def classify_candidates(df, model_path=None, window=5):
    """
    Validates detected bounces using a pre-trained Random Forest classifier.
    Candidates predicted as 'False' (0) are discarded.
    """
    # Resolve default model path
    if model_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, '..', 'assets', 'bounce_classifier.pkl')

    if not os.path.exists(model_path):
        import warnings
        warnings.warn(f"Model not found at {model_path}. Skipping classification step.")
        return df

    try:
        clf = joblib.load(model_path)
    except Exception as e:
        import warnings
        warnings.warn(f"Failed to load model: {e}")
        return df
    
    df = df.copy()
    detected_indices = df.index[df['is_bounce_detected'] == 1].tolist()
    
    if not detected_indices:
        return df
        
    # Extract features for all candidates
    X = []
    valid_indices = []
    
    for idx in detected_indices:
        features = extract_features(df, idx, window=window)
        if features is not None:
            X.append(features)
            valid_indices.append(idx)
        else:
            # Keep candidates at boundaries where features cannot be extracted
            # (Conservative approach: better a false positive than missing a real bounce)
            continue
            
    if not X:
        return df
        
    # Run inference
    predictions = clf.predict(np.array(X))
    
    # Filter out rejected candidates
    rejected_indices = [idx for idx, pred in zip(valid_indices, predictions) if pred == 0]
    
    if rejected_indices:
        df.loc[rejected_indices, 'is_bounce_detected'] = 0
            
    return df
