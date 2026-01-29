import numpy as np

from src.data_load import load_dataset_from_directory
from src.optimize import (
    optimize_parameters, optimize_parameters_2, 
    optimize_parameters_forest, optimize_parameters_de, 
    optimize_parameters_local, optimize_parameters_grid,
    optimize_with_optuna_xy, optimize_with_optuna_ang, 
    optimize_with_optuna_full
)
from src.pipeline import run_pipeline_full, run_pipeline_polar, run_pipeline_xy
from src.compute_results import compute_precision, compare_with_sample


def main():
    """
    Main function to load dataset, optimize parameters, and display results.
    Parameters meaning:
     - xvw, yvw: Window sizes for smoothing x and y velocities.
     - xaw, yaw: Window sizes for smoothing x and y accelerations.
     - avw, mvw: Window sizes for smoothing angular and module velocities.
     - aaw, maw: Window sizes for smoothing angular and module accelerations.
     - mf: Minimum distance factor to apply.
     - xat, yat: Thresholds for x and y accelerations to detect bounces
    - aat, mat: Thresholds for angular and module accelerations to detect bounces.
    - sha: Amount to shift detected bounces.
    - mew: Window size for merging detected bounces.
    - sb: Shot bounce parameter for post-processing.

    {
        "center_x_window": 2,
        "center_y_window": 2,
        "vel_x_window": 2,
        "vel_y_window": 2,
        "acc_x_window": 3,
        "acc_y_window": 2,
        "x_change_thrs": 4,
        "y_change_thrs": 9,
        "shift_amount": -3,
        "merge_window": 3,
        "tolerance": 2,
        "precision": 75.76,
        "tp": 50,
        "fp": 11,
        "fn": 5
    }


    """
    # Load dataset
    dataset_path = "datasets/2022-master-finals-fem"  # Replace with your dataset directory path
    df = load_dataset_from_directory(dataset_path, use_sample=True)

    # Best known initial point (75% precision)
    tolerance = 2
    initial_params = {
        'xw': 2, 'yw': 2, 
        'xvw': 2, 'yvw': 2, 
        'xaw': 3, 'yaw': 2,
        'mf': .5, 
        'xat': 7., 'yat': 13.,
        'sha': -3, 'mew': 3, 'sb': 0.5
    }

    # Best known initial point updated (65)
    initial_params = {'xw': 2, 'yw': 2, 'xvw': 1, 'yvw': 1, 'xaw': 2, 'yaw': 2, 'mf': 0.6350309309981311, 'xat': 7.418503052079416, 'yat': 15.0, 'sha': 0, 'mew': 4, 'sb': 0.47239465766936484}
    bounds = {
        'xw': (1, 10), 'yw': (1, 10), 
        'xvw': (1, 10), 'yvw': (1, 10), 
        'xaw': (1, 10), 'yaw': (1, 10),
        'mf': (0.1, 1.0),
        'xat': (0.1, 50.), 'yat': (0.1, 50.),
        'sha': (-12, 0), 'mew': (1, 10), 'sb': (0.2, 0.8)
    }
    max_precission = 0.0

    for i in range(10):
        best_parameters, best_precision = optimize_parameters_local(df, tolerance, initial_params, bounds)
        print(f"Iteration {i+1}: Best Precision = {best_precision:.2f} with Parameters = {best_parameters}")
        if best_precision > max_precission:
            max_precission = best_precision
            initial_params = best_parameters
            for key, value in initial_params.items():
                if key in ['xw', 'yw', 'xvw', 'yvw', 'xaw', 'yaw', 'sha', 'mew']:
                    initial_params[key] = int(value)
        else:
            break

    print("Best Parameters:", best_parameters)
    print("Best Precision:", best_precision)


def grid_seaerch():
    tolerance = 2
    initial_params = {
        'xw': 2, 'yw': 2, 
        'xvw': 2, 'yvw': 2, 
        'xaw': 3, 'yaw': 2,
        'mf': .5, 
        'xat': 7., 'yat': 13.,
        'sha': -3, 'mew': 3, 'sb': 0.5
    }
    parameter_ranges = {
        'xw': [2,3,4], 'yw': [1,2],
        'xvw': [1], 'yvw': [1],
        'xaw': [1], 'yaw': [1],
        'mf': [0.8, 0.9, 1.0],
        'xat': [6.5, 7,7.5],
        'yat': [24.75, 25, 25.25],
        'sha': [0],
        'mew': [4],
        'sb': [0.4]
    }
    dataset_path = "datasets/2022-master-finals-fem"  # Replace with your dataset directory path
    df = load_dataset_from_directory(dataset_path, use_sample=True)
    best_parameters, best_precision = optimize_parameters_grid(df, tolerance, parameter_ranges)
    print("Best Parameters from Grid Search:", best_parameters)
    print("Best Precision from Grid Search:", best_precision)


def generate_dataframe():
    initial_params = {
        'xw': 2, 'yw': 2, 
        'xvw': 2, 'yvw': 2, 
        'xaw': 3, 'yaw': 2,
        'mf': .5, 
        'xat': 7., 'yat': 13.,
        'sha': -3, 'mew': 3, 'sb': 0.5
    }

    dataset_path = "datasets/2022-master-finals-fem"  # Replace with your dataset directory path
    df = load_dataset_from_directory(dataset_path, use_sample=True)
    processed_df = run_pipeline_xy(df, **initial_params)

    results_df = compare_with_sample(processed_df, tolerance=2)
    precision = compute_precision(results_df)
    print(f"Precision of the processed dataframe: {precision:.2f}%")

    # Save the processed dataframe to a CSV file
    processed_df.to_csv("processed_dataframe.csv", index=False)
    print("Processed dataframe saved to 'processed_dataframe.csv'.")


def get_bounds(ip, limits, margin=0.5):
    bounds = {}
    for key, value in ip.items():
        if key in limits:
            low, high = limits[key]
            range_size = high - low
            bound_low = max(low, value - margin * range_size)
            bound_high = min(high, value + margin * range_size)
            if type(value) is int:
                bound_low = round(bound_low)
                bound_high = round(bound_high)
            bounds[key] = (bound_low, bound_high)
    return bounds


def optuna_xy():
    dataset_path = "datasets/2022-master-finals-fem"  # Replace with your dataset directory path
    df = load_dataset_from_directory(dataset_path, use_sample=True)

    tolerance = 1
    limits = {
        'xw': (1, 10), 'yw': (1, 10),
        'xvw': (1, 10), 'yvw': (1, 10), 
        'xaw': (1, 10), 'yaw': (1, 10),
        'mf': (0.0, 1.0),
        'dfx':(0.0, 1.0), 'dfy':(0.0, 1.0),
        'xat': (0.0, 100.), 'yat': (0.0, 100.),
        'sha': (-10, 1), 'mew': (1, 10), 'sb': (0.3, 0.6)
    }

    # Best trial: 381. Best value: 69.863 (run_pipeline_xy)  (non lineal distance factor)
    ip = {'xw': 2, 'yw': 3, 'xvw': 2, 'yvw': 2, 'xaw': 2, 'yaw': 2, 'mf': 0.8903683353815541, 'dfx': 0.8852649918298128, 'dfy': 0.9961130870844668, 'xat': 5.767120727789428, 'yat': 7.753859657878505, 'sha': -2, 'mew': 3, 'sb': 0.420843768536572}
    # Not best trial: 381. (non lineal distance factor)
    ip = {'xw': 2, 'yw': 2, 'xvw': 2, 'yvw': 2, 'xaw': 3, 'yaw': 2, 'mf': 0.5, 'dfx': 1., 'dfy': 1., 'xat': 4, 'yat': 9, 'sha': 0, 'mew': 3, 'sb': 0.4}
    # Random start result: Best value: 68.0
    ip = {'xw': 1, 'yw': 3, 'xvw': 3, 'yvw': 2, 'xaw': 2, 'yaw': 1, 'mf': 0.5061434132366975, 'dfx': 0.4734882044498823, 'dfy': 0.1464955591237385, 'xat': 7.735380356835842, 'yat': 8.994498081280947, 'sha': -1, 'mew': 2, 'sb': 0.41815064033181787}

    margin = 1.
    precision = 0
    for i in range(10):
        bounds = get_bounds(ip, limits, margin=margin)
        best_parameters, best_precision = optimize_with_optuna_xy(df, tolerance, ip, bounds, n_trials=500)
        print(f"Iteration {i+1}: Best Precision = {best_precision:.2f} with Parameters = {best_parameters}")
        
        # Calculate margin dinamically depending on the improvement by making it jump bigger if improvement is null
        if precision == best_precision:
            precision = best_precision
            ip = best_parameters
            margin = margin * 0.5
        elif best_precision > precision:
            precision = best_precision
            ip = best_parameters
            margin = margin * 0.9
        
        print(f"New margin for next iteration: {margin}")


    print("Best Parameters from Grid Search:", best_parameters)
    print("Best Precision from Grid Search:", best_precision)


def optuna_ang():
    tolerance = 2
    # First trial
    ip = {'xw': 1, 'yw': 1, 'xvw': 1, 'yvw': 1, 'avw': 1, 'mvw': 1, 'aaw': 1, 'maw':1, 'mf': 0.5, 'dfa': 0.8852649918298128, 'dfm': 0.9961130870844668, 'aat': 10., 'mat': 10., 'sha': -2, 'mew': 3, 'sb': 0.420843768536572}
    # Best trial: 503. Best value: 53.8462 (non lineal distance factor)
    ip = {'xw': 2, 'yw': 3, 'xvw': 4, 'yvw': 1, 'avw': 3, 'mvw': 1, 'aaw': 3, 'maw': 1, 'mf': 0.4612735596703449, 'dfa': 0.6884402519052708, 'dfm': 0.3330838368895238, 'aat': 14.615905693041912, 'mat': 14.132061881502667, 'sha': 1, 'mew': 6, 'sb': 0.520379366268285}
    # Best trial: 535. Best value: 59.7403 (non lineal distance factor)
    ip = {'xw': 1, 'yw': 2, 'xvw': 2, 'yvw': 2, 'avw': 1, 'mvw': 1, 'aaw': 3, 'maw': 3, 'mf': 0.9673794991177156, 'dfa': 0.8992014699390287, 'dfm': 0.7089900559671756, 'aat': 19.664352325060435, 'mat': 6.337776896542773, 'sha': -1, 'mew': 5, 'sb': 0.5483712091093551}
    # Best trial: 561. Best value: 62.3377 (non lineal distance factor)
    ip = {'xw': 1, 'yw': 1, 'xvw': 2, 'yvw': 2, 'avw': 1, 'mvw': 2, 'aaw': 3, 'maw': 3, 'mf': 0.9682176689372618, 'dfa': 0.8586586702222427, 'dfm': 0.5812267701344422, 'aat': 12.833489625387474, 'mat': 6.28804148192673, 'sha': -1, 'mew': 5, 'sb': 0.5527706342333956}
    # Best trial: 42. Best value: 63.1579 (non lineal distance factor)
    ip = {'xw': 1, 'yw': 1, 'xvw': 1, 'yvw': 1, 'avw': 1, 'mvw': 2, 'aaw': 3, 'maw': 3, 'mf': 0.963380842672425, 'dfa': 0.8711358804264588, 'dfm': 4.20753532281983, 'aat': 12.566633986560358, 'mat': 8.811716570966228, 'sha': -1, 'mew': 5, 'sb': 0.5456563152989189}
    bounds = {
        'xw': (1, 2), 'yw': (1, 2), 
        'xvw': (1, 3), 'yvw': (1, 3), 
        'avw':(1, 2), 'mvw':(1, 2), 
        'aaw':(2, 4), 'maw':(2, 4),
        'mf': (0.8, 1.0),
        'dfa':(0.8, 1.0), 'dfm':(0.4, 6.0),
        'aat':(8., 16.,), 'mat':(0.1, 10.),
        'sha': (-2, 1), 'mew': (4, 6), 'sb': (0.5, 0.6)
    }
    dataset_path = "datasets/2022-master-finals-fem"  # Replace with your dataset directory path
    df = load_dataset_from_directory(dataset_path, use_sample=True)
    best_parameters, best_precision = optimize_with_optuna_ang(df, tolerance, ip, bounds, n_trials=1000)
    print("Best Parameters from Grid Search:", best_parameters)
    print("Best Precision from Grid Search:", best_precision)


def optuna_full():
    tolerance = 2
    # Best trial: 1886. Best value: 63.5135 (non lineal distance factor)
    ip = {'xw': 1, 'yw': 1, 'xvw': 1, 'yvw': 1, 'xaw': 1, 'yaw':1, 'avw': 1, 'mvw': 1, 'aaw': 1, 'maw':1, 'mf': 0.5, 'dfx': 0.5, 'dfy':0.5, 'dfa': 0.5, 'dfm': 0.5, 'aat': 50., 'mat': 50., 'sha': -2, 'mew': 3, 'sb': 0.420843768536572}
    # Trial
    ip = {'xw': 2, 'yw': 3, 'xvw': 2, 'yvw': 2, 'avw': 2, 'xaw': 2, 'yaw': 2, 'mvw': 2, 'aaw': 3, 'maw': 3, 'mf': 0.8903683353815541, 'dfx': 0.8852649918298128, 'dfy': 0.9961130870844668, 'dfa': 0.6949090281589454, 'dfm': 0.2777971451534783, 'xat': 5.767120727789428, 'yat': 7.753859657878505, 'aat': 12.566633986560358, 'mat': 8.811716570966228, 'sha': -2, 'mew': 3, 'sb': 0.420843768536572}
    # Best trial: 1956. Best value: 66.2338 (non lineal distance factor)
    ip = {'xw': 3, 'yw': 3, 'xvw': 2, 'yvw': 2, 'avw': 2, 'xaw': 1, 'yaw': 3, 'mvw': 2, 'aaw': 3, 'maw': 4, 'mf': 0.5544147897905495, 'dfx': 0.5839762810346746, 'dfy': 0.5942227662753372, 'dfa': 0.12823025577473524, 'dfm': 0.8755692621943676, 'xat': 7.332796694842579, 'yat': 7.765475766846146, 'aat': 53.370277291959795, 'mat': 54.07114452099517, 'sha': -1, 'mew': 3, 'sb': 0.3966170437363865}
    # Best trial: 180. Best value: 66.6667 (non lineal distance factor)
    ip = {'xw': 3, 'yw': 3, 'xvw': 1, 'yvw': 2, 'avw': 3, 'xaw': 1, 'yaw': 3, 'mvw': 2, 'aaw': 3, 'maw': 4, 'mf': 0.6498245524120257, 'dfx': 0.6992319891869581, 'dfy': 0.6285405317237288, 'dfa': 0.12006096376016484, 'dfm': 0.7027901153267841, 'xat': 8.382557166617367, 'yat': 7.343751259767632, 'aat': 57.83739401626421, 'mat': 44.08863554708799, 'sha': -1, 'mew': 3, 'sb': 0.4334025654870862}
    # Best trial: 364. Best value: 67.5325 (non lineal distance factor)
    ip = {'xw': 3, 'yw': 3, 'xvw': 2, 'yvw': 2, 'avw': 4, 'xaw': 1, 'yaw': 2, 'mvw': 2, 'aaw': 3, 'maw': 4, 'mf': 0.7452205416471217, 'dfx': 0.743357111998813, 'dfy': 0.6156982194069373, 'dfa': 0.11506779065831403, 'dfm': 0.6453086764316718, 'xat': 6.5252339781638575, 'yat': 6.9518085512958985, 'aat': 57.99777763627267, 'mat': 43.159693010208194, 'sha': -2, 'mew': 3, 'sb': 0.4476185026474136}
    # Best trial: 424. Best value: 68.9189 (non lineal distance factor)
    ip = {'xw': 3, 'yw': 3, 'xvw': 2, 'yvw': 2, 'avw': 5, 'xaw': 1, 'yaw': 2, 'mvw': 1, 'aaw': 4, 'maw': 4, 'mf': 0.735499809661536, 'dfx': 0.7460476424745047, 'dfy': 0.672480977679379, 'dfa': 0.11261277547523417, 'dfm': 0.6581445229815569, 'xat': 5.9448372969422785, 'yat': 7.482734288228419, 'aat': 64.24577250520913, 'mat': 41.00449173666234, 'sha': -2, 'mew': 3, 'sb': 0.49670328547103604}
    bounds = {
        'xw': (2, 4), 'yw': (2, 4), 
        'xvw': (1, 3), 'yvw': (1, 3),
        'xaw': (1, 2), 'yaw': (1, 3),
        'avw':(3, 5), 'mvw':(1, 3), 
        'aaw':(2, 4), 'maw':(3, 5),
        'mf': (0.7, 0.8),
        'dfx':(0.7, 0.8), 'dfy':(0.5, 0.7),
        'dfa':(0.05, 0.15), 'dfm':(0.6, 0.7),
        'xat': (5., 7.5), 'yat': (6., 8.),
        'aat':(55., 65.,), 'mat':(37, 47.),
        'sha': (-3, -1), 'mew': (2, 4), 'sb': (0.4, 0.5)
    }

    dataset_path = "datasets/2022-master-finals-fem"  # Replace with your dataset directory path
    df = load_dataset_from_directory(dataset_path, use_sample=True)
    best_parameters, best_precision = optimize_with_optuna_full(df, tolerance, ip, bounds, n_trials=2000)
    print("Best Parameters from Grid Search:", best_parameters)
    print("Best Precision from Grid Search:", best_precision)



if __name__ == "__main__":
    # main()
    # generate_dataframe()
    # grid_seaerch()
    optuna_xy()
    # optuna_full()