


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
        dict: Dictionary containing precision, recall and F1-score.
    """
    tp = df['true_positive'].sum()
    fp = df['false_positive'].sum()
    fn = df['false_negative'].sum()
    precision = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    return 100 * precision

