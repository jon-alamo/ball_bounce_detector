import os
import sys

# Ensure the current directory is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bounce_detector.data_load import load_dataset_from_directory
from bounce_detector import detect_bounces
from dev_scripts.optimization.compute_results import compare_with_sample, compute_precision, compute_recall, compute_f_score

def evaluate():
    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_path, 'datasets', '2022-master-finals-fem')
    
    print(f"Loading dataset from: {dataset_path}")
    try:
        df = load_dataset_from_directory(dataset_path, use_sample=True)
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
        return

    print(f"Dataset loaded. Frames: {len(df)}")
    print("Running pipeline...")
    
    # Run the full pipeline with default parameters
    result_df = detect_bounces(df)
    
    # Compare results with ground truth
    result_df = compare_with_sample(result_df, tolerance=2)
    
    tp = result_df['true_positive'].sum()
    fp = result_df['false_positive'].sum()
    fn = result_df['false_negative'].sum()
    
    # Calculate TN: (All frames that are NOT real bounces) - False Positives
    real_bounces = result_df['is_bounce'].sum()
    total_frames = len(result_df)
    tn = (total_frames - real_bounces) - fp
    
    precision = compute_precision(result_df)
    recall = compute_recall(result_df)
    f1 = compute_f_score(result_df)
    
    print("\n" + "="*40)
    print("PIPELINE EVALUATION RESULTS")
    print("="*40)
    print(f"Dataset: {os.path.basename(dataset_path)}")
    print(f"Total Frames: {total_frames}")
    print(f"Real Bounces: {real_bounces}")
    print(f"Detected Bounces: {result_df['is_bounce_detected'].sum()}")
    print("-" * 40)
    print(f"True Positives  (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives  (TN): {tn}")
    print("-" * 40)
    print(f"Precision: {precision:.2f}%")
    print(f"Recall:    {recall:.2f}%")
    print(f"F1 Score:  {f1:.2f}%")
    print("="*40)

if __name__ == "__main__":
    evaluate()
