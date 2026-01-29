import pandas as pd
from bounce_detector.data_load import load_dataset_from_directory
from bounce_detector.pipeline.b_compute_polar import compute_polar_velocity, compute_polar_acceleration
from bounce_detector.pipeline.a_compute_xy import compute_xy_center, compute_xy_velocity

def check_stats():
    dataset_path = "datasets/2022-master-finals-fem"
    df = load_dataset_from_directory(dataset_path, use_sample=True)
    
    # Precompute necessary columns
    df = compute_xy_center(df, x_window=2, y_window=3)
    df = compute_xy_velocity(df, x_window=2, y_window=2)
    
    # Compute Polar with some default windows
    df = compute_polar_velocity(df, ang_window=2, mod_window=2)
    df = compute_polar_acceleration(df, ang_window=2, mod_window=2)
    
    print("Stats for acc_angle:")
    print(df['acc_angle'].abs().describe())
    print("\nStats for acc_module:")
    print(df['acc_module'].abs().describe())

if __name__ == "__main__":
    check_stats()