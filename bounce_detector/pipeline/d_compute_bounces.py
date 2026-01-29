import pandas as pd
import numpy as np

def compute_bounces(df, x_acc_thrs=30, y_acc_thrs=30, angle_acc_thrs=30, mod_acc_thrs=30):
    """
    Detects potential bounces by checking if acceleration exceeds dynamic thresholds.
    
    A bounce score is calculated as the ratio between acceleration and the threshold
    adjusted by the distance factors.
    """
    df['is_bounce_detected'] = False
    
    # Calculate normalized scores (Acceleration / Adjusted Threshold)
    # Using a helper dict to iterate could be cleaner, but explicit is fine for performance.
    
    s_ang = df['acc_angle'].abs() / (angle_acc_thrs * df['distance_factor_to_angle_thrs'])
    s_mod = df['acc_module'].abs() / (mod_acc_thrs * df['distance_factor_to_mod_thrs'])
    s_x = df['acc_x'].abs() / (x_acc_thrs * df['distance_factor_to_x_thrs'])
    s_y = df['acc_y'].abs() / (y_acc_thrs * df['distance_factor_to_y_thrs'])

    # Determine bounce score (maximum relative intensity across all dimensions)
    df['bounce_score'] = np.maximum.reduce([s_ang, s_mod, s_x, s_y])

    # A bounce is detected if any score > 1.0
    df['is_bounce_detected'] = df['bounce_score'] > 1.0
    return df


def compute_bounces_ang(df, angle_change_thrs=30, mod_change_thrs=30):
    """Detects bounces using only polar acceleration components."""
    df['is_bounce_detected'] = False
    
    s_ang = df['acc_angle'].abs() / (angle_change_thrs * df['distance_factor_to_angle_thrs'])
    s_mod = df['acc_module'].abs() / (mod_change_thrs * df['distance_factor_to_mod_thrs'])
    
    df['bounce_score'] = np.maximum(s_ang, s_mod)
    df['is_bounce_detected'] = df['bounce_score'] > 1.0
    return df


def compute_bounces_xy(df, x_acc_thrs=30, y_acc_thrs=30):
    """Detects bounces using only Cartesian acceleration components."""
    df['is_bounce_detected'] = False

    s_x = df['acc_x'].abs() / (x_acc_thrs * df['distance_factor_to_x_thrs'])
    s_y = df['acc_y'].abs() / (y_acc_thrs * df['distance_factor_to_y_thrs'])
    
    df['bounce_score'] = np.maximum(s_x, s_y)
    df['is_bounce_detected'] = df['bounce_score'] > 1.0
    return df
