from bounce_detector.config import Columns


def compute_xy_center(df, x_window=1, y_window=1):
    """Smooths the ball's center position coordinates using a rolling mean."""
    df[Columns.ball_x_center_col] = df[Columns.ball_x_center_col].rolling(window=x_window, min_periods=1, center=True).mean()
    df[Columns.ball_y_center_col] = df[Columns.ball_y_center_col].rolling(window=y_window, min_periods=1, center=True).mean()
    return df


def compute_xy_velocity(df, x_window=1, y_window=1):
    """Computes and smooths velocity components (first derivative of position)."""
    # Calculate raw velocity (diff)
    vx_raw = df[Columns.ball_x_center_col].diff().fillna(0)
    vy_raw = df[Columns.ball_y_center_col].diff().fillna(0)

    # Apply smoothing
    df[Columns.vel_x_col] = vx_raw.rolling(window=x_window, min_periods=1, center=True).mean()
    df[Columns.vel_y_col] = vy_raw.rolling(window=y_window, min_periods=1, center=True).mean()
    return df


def compute_xy_acceleration(df, x_window=1, y_window=1):
    """Computes and smooths acceleration components (first derivative of velocity)."""
    # Calculate raw acceleration (diff of velocity)
    ax_raw = df[Columns.vel_x_col].diff().fillna(0)
    ay_raw = df[Columns.vel_y_col].diff().fillna(0)

    # Apply smoothing
    df[Columns.acc_x_col] = ax_raw.rolling(window=x_window, min_periods=1, center=True).mean()
    df[Columns.acc_y_col] = ay_raw.rolling(window=y_window, min_periods=1, center=True).mean()
    return df
