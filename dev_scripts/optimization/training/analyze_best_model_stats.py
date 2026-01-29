import pandas as pd
from bounce_detector.data_load import load_dataset_from_directory
from bounce_detector.pipeline import run_pipeline_full, run_pipeline_xy
from bounce_detector.compute_results import compare_with_sample

def main():
    # Load data
    dataset_path = 'datasets/2022-master-finals-fem'
    df = load_dataset_from_directory(dataset_path)
    
    # Best parameters from Trial 96 (Full Pipeline)
    # Value: 67.57%
    params = {
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

    print("--- BASELINE: Thresholds Only ---")
    processed_df = run_pipeline_full(df, use_classifier=False, **params)
    results = compare_with_sample(processed_df, tolerance=3)
    
    tp = results['true_positive'].sum()
    fp = results['false_positive'].sum()
    fn = results['false_negative'].sum()
    
    precision_std = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_std = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision_std * recall_std) / (precision_std + recall_std) if (precision_std + recall_std) > 0 else 0
    
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"Precision: {precision_std:.2%}, Recall: {recall_std:.2%}, F1: {f1_score:.2%}")

    print("\n--- NEW METHOD: Two-Stage Detection (with Classifier) ---")
    processed_df_ml = run_pipeline_full(df, use_classifier=True, **params)
    results_ml = compare_with_sample(processed_df_ml, tolerance=3)
    
    tp_ml = results_ml['true_positive'].sum()
    fp_ml = results_ml['false_positive'].sum()
    fn_ml = results_ml['false_negative'].sum()
    
    precision_ml = tp_ml / (tp_ml + fp_ml) if (tp_ml + fp_ml) > 0 else 0
    recall_ml = tp_ml / (tp_ml + fn_ml) if (tp_ml + fn_ml) > 0 else 0
    f1_ml = 2 * (precision_ml * recall_ml) / (precision_ml + recall_ml) if (precision_ml + recall_ml) > 0 else 0
    
    print(f"TP: {tp_ml}, FP: {fp_ml}, FN: {fn_ml}")
    print(f"Precision: {precision_ml:.2%}, Recall: {recall_ml:.2%}, F1: {f1_ml:.2%}")
    
    improvement = f1_ml - f1_score
    print(f"\nIMPROVEMENT IN F1 SCORE: {improvement:+.2%}")

if __name__ == "__main__":
    main()
