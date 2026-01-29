from bounce_detector.config import Columns
import numpy as np


def compute_polar_velocity(df, ang_window=1, mod_window=1):
    """
    Computes angle and magnitude of velocity vectors.
    Angle is in degrees.
    """
    vx = df[Columns.vel_x_col]
    vy = df[Columns.vel_y_col]

    # Calculate polar components
    angle = np.degrees(np.arctan2(vy, vx))
    module = np.hypot(vx, vy)

    # Smooth the resulting components
    df[Columns.vel_angle_col] = angle.rolling(window=ang_window, min_periods=1, center=True).mean()
    df[Columns.vel_module_col] = module.rolling(window=mod_window, min_periods=1, center=True).mean()
    return df


def compute_polar_acceleration(df, ang_window=1, mod_window=1):
    """Computes derivatives (acceleration) of the polar velocity components."""
    # Calculate raw derivatives
    acc_angle_raw = df[Columns.vel_angle_col].diff().fillna(0)
    acc_mod_raw = df[Columns.vel_module_col].diff().fillna(0)
    
    # Smooth them
    df[Columns.acc_angle_col] = acc_angle_raw.rolling(window=ang_window, min_periods=1, center=True).mean()
    df[Columns.acc_module_col] = acc_mod_raw.rolling(window=mod_window, min_periods=1, center=True).mean()
    return df



