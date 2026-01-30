import pandas as pd

def post_process(df, shot_bounce=0.5, method='default'):
    """
    Refines detections based on game mechanics (shot categories).
    
    Rules:
    - 'Serve': Keep only the last bounce (the one before player hit). If none, guess based on ball height.
    - Other shots: If no bounce detected, infer one based on shot duration.
    """
    df = df.copy()
    df['is_bounce_detected'] = df['is_bounce_detected'].astype(int)

    # Identify changes in category to define shot sequences
    # We compare current category with previous one
    # Create group IDs for each contiguous sequence of same category
    df['shot_group'] = (df['category'] != df['category'].shift()).cumsum()

    # Iterate over groups using groupby (efficient)
    for _, group in df.groupby('shot_group'):
        category = group['category'].iloc[0]
        
        if pd.isna(category) or category == '':
            continue
            
        # Mask for this group in the main dataframe
        idx_start, idx_end = group.index[0], group.index[-1]
        
        detected_indices = group.index[group['is_bounce_detected'] == 1].tolist()
        
        # Reset detection in this region first
        df.loc[group.index, 'is_bounce_detected'] = 0
        
        if category == 'Serve':
            # Rule: Only keeping the LAST bounce in a serve sequence
            if detected_indices:
                last_bounce_idx = detected_indices[-1]
                df.at[last_bounce_idx, 'is_bounce_detected'] = 1
            else:
                # Fallback: Frame where ball is highest (min y value)
                min_y_idx = group['ball_center_y'].idxmin()
                df.at[min_y_idx, 'is_bounce_detected'] = 1
        
        else:
            # Rule: Normal shot
            if detected_indices:
                # Keep original detections
                df.loc[detected_indices, 'is_bounce_detected'] = 1
            elif method=='default':
                # Fallback: Heuristic placement based on shot duration
                offset = int(len(group) * shot_bounce)
                heuristic_idx = idx_start + offset
                # Ensure we don't go out of bounds of the group
                heuristic_idx = min(heuristic_idx, idx_end)
                df.at[heuristic_idx, 'is_bounce_detected'] = 1
            elif method == 'score':
                # Fallback: Choose frame with highest bounce score
                # Assuming 'bounce_score' column exists
                best_idx = group['bounce_score'].idxmax()
                df.at[best_idx, 'is_bounce_detected'] = 1

    
    df.drop(columns=['shot_group'], inplace=True)
    return df
