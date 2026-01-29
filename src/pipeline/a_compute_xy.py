
ball_x_center_col = 'ball_center_x'
ball_y_center_col = 'ball_center_y'


def compute_xy_center(df, x_window=1, y_window=1):
    """ Compute cartesian center positions from raw ball positions.
    """
    # Smooth the x and y positions using a rolling window
    df[ball_x_center_col] = df[ball_x_center_col].rolling(window=x_window, min_periods=1, center=True).mean()
    df[ball_y_center_col] = df[ball_y_center_col].rolling(window=y_window, min_periods=1, center=True).mean()

    return df


def compute_xy_velocity(df, x_window=1, y_window=1):
    """ Compute cartesian velocity and acceleration from cartessian positions. With peak centering.
    """
    df['vel_x'] = df[ball_x_center_col].diff().fillna(0)
    df['vel_y'] = df[ball_y_center_col].diff().fillna(0)

    # Smooth the velocity using a rolling window but centering the peaks
    df['vel_x'] = df['vel_x'].rolling(window=x_window, min_periods=1, center=True).mean()
    df['vel_y'] = df['vel_y'].rolling(window=y_window, min_periods=1, center=True).mean()

    return df


def compute_xy_acceleration(df, x_window=1, y_window=1):
    """ Compute cartesian acceleration from cartessian positions.
    """
    df['acc_x'] = df['vel_x'].diff().fillna(0)
    df['acc_y'] = df['vel_y'].diff().fillna(0)

    # Smooth the acceleration using a rolling window
    df['acc_x'] = df['acc_x'].rolling(window=x_window, min_periods=1, center=True).mean()
    df['acc_y'] = df['acc_y'].rolling(window=y_window, min_periods=1, center=True).mean()

    return df
