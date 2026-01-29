import pandas as pd
from bounce_detector.data_load import load_dataset_from_directory
from bounce_detector.pipeline import run_pipeline_xy
from bounce_detector.compute_results import compare_with_sample, compute_precision

def test_repro():
    dataset_path = "datasets/2022-master-finals-fem"
    df = load_dataset_from_directory(dataset_path, use_sample=True)
    
    # Parameters from main.py claiming 75%
    params = {
        'xw': 2, 'yw': 2, 
        'xvw': 2, 'yvw': 2, 
        'xaw': 3, 'yaw': 2,
        'mf': 1, # Default in run_pipeline_xy
        'dfx': 0, # Default in run_pipeline_xy
        'dfy': 0, # Default in run_pipeline_xy
        'xat': 4., 'yat': 9.,
        'sha': -3, 'mew': 3, 'sb': 0.5
    }
    
    # Note: dfx and dfy are 0 by default in run_pipeline_xy, so distance factor is disabled?
    # But wait, mf is passed. Even if dfx=0, compute_non_lineal_distance_factor is called.
    # If dfx=0, distance_factor_to_x_thrs = 1.0 + (dist_factor - 1.0)*0 = 1.0. 
    # So yes, distance factor has NO effect on thresholds if dfx=0.

    processed_df = run_pipeline_xy(df, **params)
    results_df = compare_with_sample(processed_df, tolerance=2)
    precision = compute_precision(results_df)

    tp = results_df['true_positive'].sum()
    fp = results_df['false_positive'].sum()
    fn = results_df['false_negative'].sum()
    
    print(f"Precision: {precision:.2f}%")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")

if __name__ == "__main__":
    test_repro()
