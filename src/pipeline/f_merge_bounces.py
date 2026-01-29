



def merge_detected_bounces(df, merge_window=5):
    """
    Merge detected bounces that are within a certain frame window.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'is_bounce_detected' column.
    merge_window (int): Number of frames within which to merge detected bounces.

    Returns:
    pd.DataFrame: DataFrame with merged 'is_bounce_detected' column.
    """
    df = df.copy()
    detected_indices = df.index[df['is_bounce_detected'] == 1].tolist()

    merged_indices = set()
    for idx in detected_indices:
        if any(abs(idx - merged_idx) <= merge_window for merged_idx in merged_indices):
            continue
        merged_indices.add(idx)

    df['is_bounce_detected'] = 0
    df.loc[list(merged_indices), 'is_bounce_detected'] = 1

    return df
