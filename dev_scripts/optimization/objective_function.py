from bounce_detector.compute_results import compute_precision, compare_with_sample, compute_f_score
from bounce_detector.pipeline import run_pipeline_full, run_pipeline_xy, run_pipeline_polar


def objective_function(df, tolerance, pipeline=run_pipeline_xy, metric='f2', **params):
    """
    Objective function to evaluate the performance of the bounce detection pipeline.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing ball position data.
    bounce_samples (list): List of ground truth bounce samples.
    metric (str): Metric to optimize ('precision', 'recall', 'f1', 'f2').
    **params: Parameters for the pipeline.

    Returns:
    float: Score of the bounce detection based on the metric.
    """

    # Run the pipeline with given parameters
    processed_df = pipeline(df, **params)

    # Compare detected bounces with ground truth samples
    matched_samples = compare_with_sample(processed_df, tolerance=tolerance)

    # Compute score
    if metric == 'precision':
        score = compute_precision(matched_samples)
    elif metric == 'f2':
        score = compute_f_score(matched_samples, beta=2)
    elif metric == 'f1':
        score = compute_f_score(matched_samples, beta=1)
    else:
        score = compute_precision(matched_samples)

    return score
