from bounce_detector.pipeline import run_pipeline_full
from bounce_detector.config import Columns


BEST_MODEL_PARAMS = {
    "xw": 2,
    "yw": 3,
    "xvw": 3,
    "yvw": 2,
    "avw": 4,
    "xaw": 1,
    "yaw": 2,
    "mvw": 3,
    "aaw": 4,
    "maw": 4,
    "mf": 0.7416,
    "dfx": 0.7651,
    "dfy": 0.892,
    "dfa": 0.8596,
    "dfm": 0.8369,
    "xat": 7.6377,
    "yat": 7.3011,
    "aat": 84.2166,
    "mat": 39.3726,
    "sha": -2,
    "mew": 3,
    "sb": 0.516
}



def detect_bounces(df, columns: dict):
    """ Detect bounces in the input dataframe by using 
    following required data with default values:
    - ball_x_center_col: int = 'ball_center_x'
    - ball_y_center_col: int = 'ball_center_y'
    - team_shot_col: str = 'team_shot'
    - has_shot_col: bool = 'has_shot'
    - category_col: str = 'category'
    - player_template_col: str = 'player_{team}_{position}_{axis}'

    To override default column names, provide a dictionary with 
    the column names to be used.

    The function adds the following columns to the dataframe
    which also can be customized via the `columns` parameter:
    - vel_x_col: float  = 'vel_x'
    - vel_y_col: float = 'vel_y'
    - acc_x_col: float = 'acc_x'
    - acc_y_col: float = 'acc_y'
    - vel_angle_col: float  = 'vel_angle'
    - vel_module_col: float  = 'vel_module'
    - acc_angle_col: float  = 'acc_angle'
    - acc_module_col: float  = 'acc_module'
    - is_bounce_detected_col: bool = 'is_bounce_detected'


    Parameters:
        df (pd.DataFrame): Input DataFrame containing ball position data.
        columns (dict): Dictionary mapping required column keys to actual column names in df.
    """
    for key, name in columns.items():
        setattr(Columns, key, name)    
    df = run_pipeline_full(df, **BEST_MODEL_PARAMS)
    return df
