


def post_process(df, shot_bounce=0.5):
    """ Post process the Dataframe df for sequences where category has the 
    same value not being this value empty. For those sequences, depending on 
    the value of the category itself:
        - If "category" is "Serve" for the sequence and there are detected bounces,
          keep only the last detected bounce.
        - If category is "Serve" but no boubce where detected, set the frame with the highest
          "ball_height" during the shot as a detected bounce.
        - If category is not "Serve" and there are detected bounces, keep as it is.
        - If category is not "Serve" and no bounce were detected, set as detected bounce 
          the frame given by shot_bounce fraction of the shot duration (e.g., 0.5 for middle frame).
    """
    df = df.copy()
    df['is_bounce_detected'] = df['is_bounce_detected'].astype(int)

    change_indices = df.index[df['category'] != df['category'].shift(1, fill_value='')].tolist()
    change_indices.append(len(df))

    is_true = 1
    is_false = 0

    for i in range(len(change_indices) - 1):
        start_idx = change_indices[i]
        end_idx = change_indices[i + 1]
        category = df['category'].iloc[start_idx]

        if category == '':
            continue

        shot_df = df.loc[start_idx:end_idx - 1]
        detected_bounces = shot_df.index[shot_df['is_bounce_detected'] == 1].tolist()

        if category == 'Serve':
            if detected_bounces:
                last_bounce = detected_bounces[-1]
                df.loc[start_idx:end_idx - 1, 'is_bounce_detected'] = is_false
                df.at[last_bounce, 'is_bounce_detected'] = is_true
            else:
                max_height_frame = shot_df['ball_center_y'].idxmax()
                df.loc[start_idx:end_idx - 1, 'is_bounce_detected'] = is_false
                df.at[max_height_frame, 'is_bounce_detected'] = is_true
        else:
            if not detected_bounces:
                shot_length = end_idx - start_idx
                bounce_frame_offset = int(shot_length * shot_bounce)
                bounce_frame = start_idx + bounce_frame_offset
                df.loc[start_idx:end_idx - 1, 'is_bounce_detected'] = is_false
                df.at[bounce_frame, 'is_bounce_detected'] = is_true

    return df
