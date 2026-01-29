def compare_with_sample(df, tolerance=1):
    """ Compare detected bounces (is_detected_bounce) with real sample bounces (is_bounce) 
    within a tolerance window to get true positives, false positives and false negatives.
    Args:
        df (pd.DataFrame): DataFrame containing 'is_bounce' and 'is_bounce_detected' columns.
        tolerance (int): Number of frames to consider for matching bounces.
    """
    df = df.copy()
    df['true_positive'] = 0
    df['false_positive'] = 0
    df['false_negative'] = 0

    real_bounce_frames = df.index[df['is_bounce'] == 1].tolist()
    detected_bounce_frames = df.index[df['is_bounce_detected'] == 1].tolist()

    matched_real = set()
    matched_detected = set()

    for real_frame in real_bounce_frames:
        for detected_frame in detected_bounce_frames:
            if abs(real_frame - detected_frame) <= tolerance:
                df.at[detected_frame, 'true_positive'] = 1
                matched_real.add(real_frame)
                matched_detected.add(detected_frame)
                break

    for detected_frame in detected_bounce_frames:
        if detected_frame not in matched_detected:
            df.at[detected_frame, 'false_positive'] = 1

    for real_frame in real_bounce_frames:
        if real_frame not in matched_real:
            df.at[real_frame, 'false_negative'] = 1

    return df


def compute_precision(df):
    """ Compute precision, recall and F1-score based on true positives, false positives and false negatives.
    Args:
        df (pd.DataFrame): DataFrame containing 'true_positive', 'false_positive' and 'false_negative' columns.
    Returns:
        float: The Jaccard index (TP / (TP + FP + FN)) percentage.
    """
    tp = df['true_positive'].sum()
    fp = df['false_positive'].sum()
    fn = df['false_negative'].sum()
    precision = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    return 100 * precision

def compute_recall(df):
    """ Compute recall based on true positives and false negatives.
    Args:
        df (pd.DataFrame): DataFrame containing 'true_positive' and 'false_negative' columns.
    Returns:
        float: Recall percentage.
    """
    tp = df['true_positive'].sum()
    fn = df['false_negative'].sum()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 100 * recall

def compute_f_score(df, beta=1):
    """ Compute F-beta score.
    Args:
        df (pd.DataFrame): DataFrame containing TP, FP, FN.
        beta (float): Weight of recall in the harmonic mean. beta=1 is F1, beta=2 weighs recall higher.
    Returns:
        float: F-beta score percentage.
    """
    tp = df['true_positive'].sum()
    fp = df['false_positive'].sum()
    fn = df['false_negative'].sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
        
    f_score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    return 100 * f_score

