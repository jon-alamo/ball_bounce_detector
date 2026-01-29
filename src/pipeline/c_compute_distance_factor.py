import pandas as pd
import numpy as np


def compute_distance_factor_for_threshold_type(df, df_to_x_thrs=1.0, df_to_y_thrs=1.0, df_to_angle_thrs=1.0, df_to_mod_thrs=1.0):
    """ Calcula el factor de distancia a aplicar a cada uno de los 
    umbrales de manera que si por ejemplo factor_to_x_thrs=0, el 
    umbral de x no se verá afectado por el factor de distancia,
    mientras que si factor_to_x_thrs=1.0, el umbral de x se verá
    totalmente afectado por el factor de distancia. Así para cada 
    uno de los umbrales y las variables de entrada de este función.
    Parameters:
        df (pd.DataFrame): Input DataFrame containing ball position data.
        factor_to_x_thrs (float): Weight of distance factor to apply to x threshold.
        factor_to_y_thrs (float): Weight of distance factor to apply to y threshold.
        factor_to_angle_thrs (float): Weight of distance factor to apply to angle threshold.
        factor_to_mod_thrs (float): Weight of distance factor to apply to module threshold.
    """
    df = df.copy()
    df['distance_factor_to_x_thrs'] = 1.0 + (df['distance_factor'] - 1.0) * df_to_x_thrs
    df['distance_factor_to_y_thrs'] = 1.0 + (df['distance_factor'] - 1.0) * df_to_y_thrs
    df['distance_factor_to_angle_thrs'] = 1.0 + (df['distance_factor'] - 1.0) * df_to_angle_thrs
    df['distance_factor_to_mod_thrs'] = 1.0 + (df['distance_factor'] - 1.0) * df_to_mod_thrs
    return df


def compute_distance_factor(df, min_factor=0.5, df_to_x_thrs=1.0, df_to_y_thrs=1.0, df_to_angle_thrs=1.0, df_to_mod_thrs=1.0):
    """ Calcula el factor de distancia a aplicar en función del punto
    más bajo de la bbox del jugador que golpea la bola en el momento 
    del golpeo y de los valores mínimos y máximos de los valores más 
    bajos de los bbox de los jugadores en todo el dataset. Para ello, 
    hay que determinar por un lado la y mínima y máxima en píxeles a 
    partir de las columnas player_a_left_y, player_a_left_h, 
    player_a_drive_y, player_a_drive_h ... como el máximo y mínimo del 
    valor: "player_<a/b>_<left/drive>_y + player_<a/b>_<left/drive>_h".
    Esto indicará el rango completo en píxeles. En su mínimo el factor de 
    distancia será min_factor y en su máximo 1.0, siendo lineal la 
    interpolación entre ambos puntos.
    Luego, se calculará para cada golpeo, en el frame intermedio de la 
    secuencia de golpeo (secuencias de valores iguales de "category" no vacías)
    la posición inferior de la bbox del jugador que golpea la bola teniendo en 
    cuenta el equipo que ejecuta el golpe para saber si el jugador está abajo o 
    arriba y la posición de la bola para ver cuál de los dos del equipo está más 
    cerca de la bola, determinando así quién ejecuta el golpe. Para ello hay que 
    tener en cuenta las columnas:
        - team_shot: 'a' o 'b'
        - ball_center_x: posición x de la bola en píxeles
        - ball_center_y: posición y de la bola en píxeles
        - player_<team_shot>_left_x: posición x del jugador left del equipo que golpea
        - player_<team_shot>_left_y: posición y del jugador left del equipo que golpea
        - player_<team_shot>_left_h: altura del jugador left del equipo que golpea
        - player_<team_shot>_drive_x: posición x del jugador drive del equipo que golpea
        - player_<team_shot>_drive_y: posición y del jugador drive del equipo que golpea
        - player_<team_shot>_drive_h: altura del jugador drive del equipo que golpea
    Luego con la posición inferior del jugador que golpea (y+h), se calculará
    el factor de distancia linealmente entre min_factor y 1.0 según el rango
    determinado anteriormente y se anotará para el frame intermedio a la secuencia del 
    golpeo en cuestión en una nueva columna 'distance_factor'.
    Luego, se hará una interpolación lineal para rellenar los valores de los frames 
    intermedios entre golpeos.
    Parameters:
        df (pd.DataFrame): Input DataFrame containing ball position data.
        min_factor (float): Minimum distance factor to apply.        
    """
    df = df.copy()

    if min_factor < 0.0 or min_factor > 1.0:
        raise ValueError("min_factor must be between 0.0 and 1.0")
    elif min_factor == 1.0:
        df['distance_factor'] = 1.0
        return df

    # Determine global min and max y positions of players' feet
    player_bottoms = []
    for team in ['a', 'b']:
        for position in ['left', 'drive']:
            bottom_y = df[f'player_{team}_{position}_y'] + df[f'player_{team}_{position}_h']
            player_bottoms.append(bottom_y)
    all_bottoms = pd.concat(player_bottoms, axis=1)
    global_min_y = all_bottoms.min().min()
    global_max_y = all_bottoms.max().max()

    def calculate_distance_factor(row):
        if pd.isna(row['team_shot']):
            return np.nan
        
        team = row['team_shot']
        ball_x = row['ball_center_x']
        ball_y = row['ball_center_y']

        left_x = row[f'player_{team}_left_x']
        left_y = row[f'player_{team}_left_y']
        left_h = row[f'player_{team}_left_h']
        left_bottom_y = left_y + left_h

        drive_x = row[f'player_{team}_drive_x']
        drive_y = row[f'player_{team}_drive_y']
        drive_h = row[f'player_{team}_drive_h']
        drive_bottom_y = drive_y + drive_h

        # Determine which player is closer to the ball
        left_distance = np.sqrt((ball_x - left_x)**2 + (ball_y - left_y)**2)
        drive_distance = np.sqrt((ball_x - drive_x)**2 + (ball_y - drive_y)**2)

        if left_distance < drive_distance:
            player_bottom_y = left_bottom_y
        else:
            player_bottom_y = drive_bottom_y

        # Calculate distance factor
        if player_bottom_y <= global_min_y:
            return min_factor
        elif player_bottom_y >= global_max_y:
            return 1.0
        else:
            return min_factor + (1.0 - min_factor) * ((player_bottom_y - global_min_y) / (global_max_y - global_min_y))

    # Identify shot sequences and calculate distance factors at mid frames
    df['distance_factor'] = np.nan
    shot_groups = df['category'].notna().cumsum()
    for _, group in df.groupby(shot_groups):
        if group['category'].iloc[0] is not np.nan:
            mid_idx = group.index[len(group) // 2]
            df.at[mid_idx, 'distance_factor'] = calculate_distance_factor(df.loc[mid_idx])
    # Interpolate distance factors for intermediate frames
    df['distance_factor'] = df['distance_factor'].interpolate(method='linear')

    df = compute_distance_factor_for_threshold_type(df, df_to_x_thrs=df_to_x_thrs, df_to_y_thrs=df_to_y_thrs, df_to_angle_thrs=df_to_angle_thrs, df_to_mod_thrs=df_to_mod_thrs)

    return df



def compute_non_lineal_distance_factor(df, min_factor=0.5, df_to_x_thrs=1.0, df_to_y_thrs=1.0, df_to_angle_thrs=1.0, df_to_mod_thrs=1.0):
    """
    Similar to compute_distance_factor but using a non-linear mapping, but using a curve that gives more weight to the extreme values closer
    to the moments of the hit rather than a linear interpolation.
    Parameters:
        df (pd.DataFrame): Input DataFrame containing ball position data.
        min_factor (float): Minimum distance factor to apply.        
    """
    df = df.copy()

    if min_factor < 0.0 or min_factor > 1.0:
        raise ValueError("min_factor must be between 0.0 and 1.0")
    elif min_factor == 1.0:
        df['distance_factor'] = 1.0
        return df

    # Determine global min and max y positions of players' feet
    player_bottoms = []
    for team in ['a', 'b']:
        for position in ['left', 'drive']:
            bottom_y = df[f'player_{team}_{position}_y'] + df[f'player_{team}_{position}_h']
            player_bottoms.append(bottom_y)
    all_bottoms = pd.concat(player_bottoms, axis=1)
    global_min_y = all_bottoms.min().min()
    global_max_y = all_bottoms.max().max()

    def calculate_distance_factor(row):
        if pd.isna(row['team_shot']):
            return np.nan
        
        team = row['team_shot']
        ball_x = row['ball_center_x']
        ball_y = row['ball_center_y']

        left_x = row[f'player_{team}_left_x']
        left_y = row[f'player_{team}_left_y']
        left_h = row[f'player_{team}_left_h']
        left_bottom_y = left_y + left_h

        drive_x = row[f'player_{team}_drive_x']
        drive_y = row[f'player_{team}_drive_y']
        drive_h = row[f'player_{team}_drive_h']
        drive_bottom_y = drive_y + drive_h

        # Determine which player is closer to the ball
        left_distance = np.sqrt((ball_x - left_x)**2 + (ball_y - left_y)**2)
        drive_distance = np.sqrt((ball_x - drive_x)**2 + (ball_y - drive_y)**2)

        if left_distance < drive_distance:
            player_bottom_y = left_bottom_y
        else:
            player_bottom_y = drive_bottom_y

        # Calculate distance factor using a non-linear mapping (quadratic)
        if player_bottom_y <= global_min_y:
            return min_factor
        elif player_bottom_y >= global_max_y:
            return 1.0
        else:
            normalized_position = (player_bottom_y - global_min_y) / (global_max_y - global_min_y)
            non_linear_factor = min_factor + (1.0 - min_factor) * (normalized_position ** 2)
            return non_linear_factor
    # Identify shot sequences and calculate distance factors at mid frames
    df['distance_factor'] = np.nan
    shot_groups = df['category'].notna().cumsum()
    for _, group in df.groupby(shot_groups):
        if group['category'].iloc[0] is not np.nan:
            mid_idx = group.index[len(group) // 2]
            df.at[mid_idx, 'distance_factor'] = calculate_distance_factor(df.loc[mid_idx])
    # Interpolate distance factors for intermediate frames
    df['distance_factor'] = df['distance_factor'].interpolate(method='linear')
    df = compute_distance_factor_for_threshold_type(df, df_to_x_thrs=df_to_x_thrs, df_to_y_thrs=df_to_y_thrs, df_to_angle_thrs=df_to_angle_thrs, df_to_mod_thrs=df_to_mod_thrs)
    return df