



def shift_detected_bounces(df, shift_amount=0):
    """ Shift detected bounces (is_bounce_detected) by a certain number of frames.
    Args:
        df (pd.DataFrame): DataFrame containing 'is_bounce_detected' column.
        shift_amount (int): Number of frames to shift. Positive values shift forward, negative values shift backward.
    Returns:
        pd.DataFrame: DataFrame with shifted 'is_bounce_detected' column.
    """
    df = df.copy()
    df['is_bounce_detected'] = df['is_bounce_detected'].shift(shift_amount, fill_value=0)
    return df

