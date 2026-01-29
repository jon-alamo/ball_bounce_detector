def shift_detected_bounces(df, shift_amount=0):
    """
    Shifts the detection markers by a specified number of frames.
    Useful to align detection with the actual visual impact if lag exists.
    """
    if shift_amount == 0:
        return df
        
    df = df.copy()
    df['is_bounce_detected'] = df['is_bounce_detected'].shift(shift_amount, fill_value=0)
    
    if 'bounce_score' in df.columns:
        df['bounce_score'] = df['bounce_score'].shift(shift_amount, fill_value=0)
        
    return df

