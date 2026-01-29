from bounce_detector.pipeline.a_compute_xy import compute_xy_center, compute_xy_velocity, compute_xy_acceleration
from bounce_detector.pipeline.b_compute_polar import compute_polar_velocity, compute_polar_acceleration
from bounce_detector.pipeline.c_compute_distance_factor import compute_distance_factor, compute_non_lineal_distance_factor
from bounce_detector.pipeline.d_compute_bounces import compute_bounces, compute_bounces_xy, compute_bounces_ang
from bounce_detector.pipeline.e_shift_bounces import shift_detected_bounces
from bounce_detector.pipeline.f_merge_bounces import merge_detected_bounces
from bounce_detector.pipeline.g_post_process import post_process
from bounce_detector.pipeline.h_classify_candidates import classify_candidates



def run_pipeline_xy(df, xw=2, yw=2, xvw=2, yvw=2, xaw=3, yaw=2, mf=1, dfx=0, dfy=0, xat=4, yat=9, sha=-3, mew=3, sb=0.5):
    """ Run the full data processing pipeline on the input dataframe.
    Parameters:
        df (pd.DataFrame): Input DataFrame containing ball position data.
        xw (int): Window size for smoothing x center position.
        yw (int): Window size for smoothing y center position.
        xvw (int): Window size for smoothing x velocity.
        yvw (int): Window size for smoothing y velocity.
        xaw (int): Window size for smoothing x acceleration.
        yaw (int): Window size for smoothing y acceleration.
        mf (float): Minimum distance factor to apply.
        xat (float): Threshold for x acceleration to detect bounces.
        yat (float): Threshold for y acceleration to detect bounces.
        sha (int): Amount to shift detected bounces.
        mew (int): Window size for merging detected bounces.
        sb (float): Shot bounce parameter for post-processing.
    Returns:
        pd.DataFrame: Processed DataFrame with bounce detection.
    """
    temp_df = df.copy()
    temp_df = compute_xy_center(temp_df, x_window=xw, y_window=yw)
    temp_df = compute_xy_velocity(temp_df, x_window=xvw, y_window=yvw)
    temp_df = compute_xy_acceleration(temp_df, x_window=xaw, y_window=yaw)
    temp_df = compute_non_lineal_distance_factor(temp_df, min_factor=mf, df_to_x_thrs=dfx, df_to_y_thrs=dfy)
    temp_df = compute_bounces_xy(temp_df, x_acc_thrs=xat, y_acc_thrs=yat)
    temp_df = shift_detected_bounces(temp_df, shift_amount=sha)
    temp_df = merge_detected_bounces(temp_df, merge_window=mew)
    temp_df = post_process(temp_df, shot_bounce=sb)
    return temp_df


def run_pipeline_polar(df, xw, yw, xvw, yvw, avw=1, mvw=1, aaw=1, maw=1, mf=0.5, dfa=1, dfm=1, aat=30, mat=30, sha=0, mew=3, sb=0.5):
    """ Run the full data processing pipeline on the input dataframe.
    Parameters:
        df (pd.DataFrame): Input DataFrame containing ball position data.
        avw (int): Window size for smoothing angular velocity.
        mvw (int): Window size for smoothing module velocity.
        aaw (int): Window size for smoothing angular acceleration.
        maw (int): Window size for smoothing module acceleration.
        mf (float): Minimum distance factor to apply.
        aat (float): Threshold for angular acceleration to detect bounces.
        mat (float): Threshold for module acceleration to detect bounces.
        sha (int): Amount to shift detected bounces.
        mew (int): Window size for merging detected bounces.
        sb (float): Shot bounce parameter for post-processing.
    Returns:
        pd.DataFrame: Processed DataFrame with bounce detection.
    """
    temp_df = df.copy()
    temp_df = compute_xy_center(temp_df, x_window=xw, y_window=yw)
    temp_df = compute_xy_velocity(temp_df, x_window=xvw, y_window=yvw)
    temp_df = compute_polar_velocity(temp_df, ang_window=avw, mod_window=mvw)
    temp_df = compute_polar_acceleration(temp_df, ang_window=aaw, mod_window=maw)
    temp_df = compute_non_lineal_distance_factor(temp_df, min_factor=mf, df_to_angle_thrs=dfa, df_to_mod_thrs=dfm)
    temp_df = compute_bounces_ang(temp_df, angle_change_thrs=aat, mod_change_thrs=mat)    
    temp_df = shift_detected_bounces(temp_df, shift_amount=sha)
    temp_df = merge_detected_bounces(temp_df, merge_window=mew)
    temp_df = post_process(temp_df, shot_bounce=sb)
    return temp_df





def run_pipeline_full(df, xw=1, yw=1, xvw=1, yvw=1, xaw=1, yaw=1, avw=1, mvw=1, aaw=1, maw=1, mf=0.5, dfx=1, dfy=1, dfa=1, dfm=1, xat=30, yat=30, aat=30, mat=30, sha=0, mew=3, sb=0.5, use_classifier=False):
    """ Run the full data processing pipeline on the input dataframe.
    Parameters:
        df (pd.DataFrame): Input DataFrame containing ball position data.
        xvw (int): Window size for smoothing x velocity.
        yvw (int): Window size for smoothing y velocity.
        xaw (int): Window size for smoothing x acceleration.
        yaw (int): Window size for smoothing y acceleration.
        avw (int): Window size for smoothing angular velocity.
        mvw (int): Window size for smoothing module velocity.
        aaw (int): Window size for smoothing angular acceleration.
        maw (int): Window size for smoothing module acceleration.
        mf (float): Minimum distance factor to apply.
        xat (float): Threshold for x acceleration to detect bounces.
        yat (float): Threshold for y acceleration to detect bounces.
        aat (float): Threshold for angular acceleration to detect bounces.
        mat (float): Threshold for module acceleration to detect bounces.
        sha (int): Amount to shift detected bounces.
        mew (int): Window size for merging detected bounces.
        sb (float): Shot bounce parameter for post-processing.
        use_classifier (bool): Whether to use the ML classifier to filter false positives.
    Returns:
        pd.DataFrame: Processed DataFrame with bounce detection.
    """
    temp_df = df.copy()
    temp_df = compute_xy_center(temp_df, x_window=xw, y_window=yw)
    temp_df = compute_xy_velocity(temp_df, x_window=xvw, y_window=yvw)
    temp_df = compute_xy_acceleration(temp_df, x_window=xaw, y_window=yaw)
    temp_df = compute_polar_velocity(temp_df, ang_window=avw, mod_window=mvw)
    temp_df = compute_polar_acceleration(temp_df, ang_window=aaw, mod_window=maw)
    temp_df = compute_non_lineal_distance_factor(temp_df, min_factor=mf, df_to_x_thrs=dfx, df_to_y_thrs=dfy, df_to_angle_thrs=dfa, df_to_mod_thrs=dfm)
    temp_df = compute_bounces(temp_df, x_acc_thrs=xat, y_acc_thrs=yat, angle_acc_thrs=aat, mod_acc_thrs=mat)    
    temp_df = shift_detected_bounces(temp_df, shift_amount=sha)
    temp_df = merge_detected_bounces(temp_df, merge_window=mew)
    if use_classifier:
        temp_df = classify_candidates(temp_df)
    temp_df = post_process(temp_df, shot_bounce=sb)
    return temp_df
