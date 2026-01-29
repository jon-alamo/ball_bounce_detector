def merge_detected_bounces(df, merge_window=5):
    """
    Consolidates multiple consecutive detections into a single event.
    Uses 'bounce_score' to pick the strongest candidate (Non-Maximum Suppression).
    """
    if 'is_bounce_detected' not in df.columns or df['is_bounce_detected'].sum() == 0:
        return df

    df = df.copy()
    detected_indices = df.index[df['is_bounce_detected'] == 1].tolist()

    final_indices = []

    if 'bounce_score' in df.columns:
        # Non-Maximum Suppression based on score
        # 1. Get all candidates and their scores
        candidates = df.loc[detected_indices, 'bounce_score'].sort_values(ascending=False)
        
        pool = candidates.index.tolist()
        while pool:
            best_idx = pool.pop(0) # Highest score remaining
            final_indices.append(best_idx)
            
            # Remove all neighbors within window
            pool = [idx for idx in pool if abs(idx - best_idx) > merge_window]
    else:
        # Fallback: keep earliest index, ignore subsequent ones within window
        processed = set()
        for idx in detected_indices:
            if idx in processed:
                continue
                
            final_indices.append(idx)
            # Mark neighbors as processed
            # (Crude loop, but sufficient for small N of detections)
            # A more efficient way would be iterating sorted indices.
            pass # Actually, logic below is simpler:
            
        # Re-implementation of simple linear merge
        final_indices = []
        last_idx = -merge_window - 1
        for idx in sorted(detected_indices):
            if idx - last_idx > merge_window:
                final_indices.append(idx)
                last_idx = idx

    # Reconstruct the column
    df['is_bounce_detected'] = 0
    df.loc[final_indices, 'is_bounce_detected'] = 1

    return df
