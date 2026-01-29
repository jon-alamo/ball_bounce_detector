import numpy as np


def compute_polar_velocity(df, ang_window=1, mod_window=1):
    """ Compute polar velocity from cartesian velocity.
    """
    df['vel_angle'] = np.arctan2(df['vel_y'], df['vel_x']) * 180 / np.pi    
    df['vel_module'] = np.sqrt(df['vel_x']**2 + df['vel_y']**2)

    # Smooth the angle using a rolling window
    df['vel_angle'] = df['vel_angle'].rolling(window=ang_window, min_periods=1, center=True).mean()
    df['vel_module'] = df['vel_module'].rolling(window=mod_window, min_periods=1, center=True).mean()
    return df


def compute_polar_acceleration(df, ang_window=1, mod_window=1):
    """ Compute polar velocity from cartesian acceleration.
    """
    df['acc_angle'] = df['vel_angle'].diff().fillna(0)
    df['acc_module'] = df['vel_module'].diff().fillna(0)
    
    # Smooth the polar acceleration using a rolling window
    df['acc_angle'] = df['acc_angle'].rolling(window=ang_window, min_periods=1, center=True).mean()
    df['acc_module'] = df['acc_module'].rolling(window=mod_window, min_periods=1, center=True).mean()
    return df



