import optuna
import pandas as pd
from bounce_detector.data_load import load_dataset_from_directory
from bounce_detectoroptimize import optimize_with_optuna_xy, optimize_with_optuna_full

def run_optimization():
    dataset_path = "datasets/2022-master-finals-fem"
    df = load_dataset_from_directory(dataset_path, use_sample=True)
    
    # Best parameters from previous run (Trial 96) - Precision ~69%
    initial_params = {
        'xw': 2, 'yw': 3, 
        'xvw': 3, 'yvw': 2, 
        'xaw': 1, 'yaw': 2, 
        'avw': 4, 'mvw': 3,
        'aaw': 4, 'maw': 4,
        'mf': 0.7416, 
        'dfx': 0.7651, 'dfy': 0.8920, 
        'dfa': 0.8596, 'dfm': 0.8369,
        'xat': 7.6377, 'yat': 7.3011, 
        'aat': 84.2166, 'mat': 39.3726,
        'sha': -2, 'mew': 3, 'sb': 0.5160
    }

    # Narrowed bounds for fine-tuning
    bounds = {
        'xw': (1, 3), 'yw': (2, 4),
        'xvw': (2, 4), 'yvw': (1, 3), 
        'xaw': (1, 3), 'yaw': (1, 3),
        'avw': (3, 5), 'mvw': (2, 4),
        'aaw': (3, 5), 'maw': (3, 5),
        'mf': (0.6, 0.9),
        'dfx': (0.6, 0.9), 'dfy': (0.8, 1.0),
        'dfa': (0.7, 1.0), 'dfm': (0.7, 1.0),
        'xat': (6.0, 9.0), 'yat': (6.0, 9.0),
        'aat': (60.0, 100.0), 'mat': (30.0, 50.0),
        'sha': (-3, -1), 'mew': (3, 8), 'sb': (0.45, 0.6)
    }
    
    print("Starting Fine-Tuning Optimization (XY + Polar)...")
    # Increased tolerance to 3 to match evaluation standard
    best_params, best_value = optimize_with_optuna_full(df, tolerance=3, initial_params=initial_params, bounds=bounds, n_trials=50) # Reduced trials for demo/iteration
    print(f"Best F2-Score: {best_value}")
    print(f"Best Params: {best_params}")

if __name__ == "__main__":
    run_optimization()
