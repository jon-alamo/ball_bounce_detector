import pandas as pd
from bounce_detector.data_load import load_dataset_from_directory
from bounce_detector.pipeline import run_pipeline_full
from bounce_detector.compute_results import compare_with_sample

def main():
    # --- Configuration ---
    DATASET_PATH = "datasets/2022-master-finals-fem"
    TOLERANCE = 3    # Frames tolerance for matching
    OUTPUT_FILE = "ml_pipeline_results.csv"

    # Best "High Recall" Parameters used to train the model
    # These must be the same parameters used during training to ensure features match
    PARAMS = {
        'xw': 2, 'yw': 3,
        'xvw': 3, 'yvw': 2,
        'avw': 4,
        'xaw': 1, 'yaw': 2,
        'mvw': 2,
        'aaw': 4, 'maw': 4,
        'mf': 0.7943415504795227,
        'dfx': 0.7531188176465113,
        'dfy': 0.8104011259381378,
        'dfa': 0.8232777159386675,
        'dfm': 0.8194713844008937,
        'xat': 6.253175940169116,
        'yat': 8.81262256777441,
        'aat': 94.8669164127763,
        'mat': 46.80580250147934,
        'sha': -2,
        'mew': 4,
        'sb': 0.4547560199076277
    }

    print(f"Loading dataset from: {DATASET_PATH}")
    df = load_dataset_from_directory(DATASET_PATH, use_sample=True)

    print("\nRunning Full Pipeline with ML Classifier...")
    # enable use_classifier=True to filter candidates with the Random Forest model
    processed_df = run_pipeline_full(df, use_classifier=True, **PARAMS)

    # Evaluate results
    print("\nEvaluating performance...")
    results = compare_with_sample(processed_df, tolerance=TOLERANCE)
    
    tp = results['true_positive'].sum()
    fp = results['false_positive'].sum()
    fn = results['false_negative'].sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("-" * 30)
    print("FINAL RESULTS")
    print("-" * 30)
    print(f"True Positives (Correct):   {tp}")
    print(f"False Positives (Ghosts):   {fp}")
    print(f"False Negatives (Missed):   {fn}")
    print("-" * 30)
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 Score:  {f1:.2%}")
    print("-" * 30)

    # Export results
    print(f"Saving detected bounces to {OUTPUT_FILE}...")
    output_cols = ['frame', 'ball_center_x', 'ball_center_y', 'is_bounce_detected', 'true_positive', 'false_positive']
    results[results['is_bounce_detected'] == 1][output_cols].to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
