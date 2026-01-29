

def compute_bounces(df, x_acc_thrs=30, y_acc_thrs=30, angle_acc_thrs=30, mod_acc_thrs=30):
    """ Compute bounce based on sudden changes in angle/module acceleration and x/y acceleration.
    Args:
        df (pd.DataFrame): DataFrame containing at least 'acc_angle', 'acc_module', 'acc_x' and 'acc_y'
    Returns:
        pd.DataFrame: DataFrame with an additional column 'is_bounce_detected' indicating detected bounces.

    """
    df['is_bounce_detected'] = False
    bounce_conditions = (
        (df['acc_angle'].abs() > angle_acc_thrs * df['distance_factor_to_angle_thrs']) |
        (df['acc_module'].abs() > mod_acc_thrs * df['distance_factor_to_mod_thrs']) |
        (df['acc_x'].abs() > x_acc_thrs * df['distance_factor_to_x_thrs']) |
        (df['acc_y'].abs() > y_acc_thrs * df['distance_factor_to_y_thrs'])
    )
    df.loc[bounce_conditions, 'is_bounce_detected'] = True
    return df



def compute_bounces_ang(df, angle_change_thrs=30, mod_change_thrs=30):
    """ Compute bounce based on sudden changes in angle and module acceleration given by acc_angle and acc_module.
    Args:
        df (pd.DataFrame): DataFrame containing at least 'acc_angle' and 'acc_module'
    Returns:
        pd.DataFrame: DataFrame with an additional column 'is_bounce_detected' indicating detected bounces.

    """
    df['is_bounce_detected'] = False
    bounce_conditions = (
        (df['acc_angle'].abs() > (angle_change_thrs * df['distance_factor_to_angle_thrs'])) | 
        (df['acc_module'].abs() > (mod_change_thrs * df['distance_factor_to_mod_thrs']))
    )
    df.loc[bounce_conditions, 'is_bounce_detected'] = True
    return df



def compute_bounces_xy(df, x_acc_thrs=30, y_acc_thrs=30):
    """ Compute bounce based on sudden changes in x and y acceleration given by acc_x and acc_y.
    Args:
        df (pd.DataFrame): DataFrame containing at least 'acc_x' and 'acc_y'
    Returns:
        pd.DataFrame: DataFrame with an additional column 'is_bounce_detected' indicating detected bounces.
    """
    df['is_bounce_detected'] = False
    bounce_conditions = (
        (df['acc_x'].abs() > x_acc_thrs * df['distance_factor_to_x_thrs']) |
        (df['acc_y'].abs() > y_acc_thrs * df['distance_factor_to_y_thrs'])
    )
    df.loc[bounce_conditions, 'is_bounce_detected'] = True
    return df
