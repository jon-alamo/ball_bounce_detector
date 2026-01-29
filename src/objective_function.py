from src.compute_results import compute_precision, compare_with_sample
from src.pipeline import run_pipeline_xy, run_pipeline_full
from src.compute_results import compute_precision, compare_with_sample
from src.pipeline import run_pipeline_full, run_pipeline_xy, run_pipeline_polar


def objective_function(df, tolerance, pipeline=run_pipeline_xy, **params):
    """
    Objective function to evaluate the performance of the bounce detection pipeline.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing ball position data.
    bounce_samples (list): List of ground truth bounce samples.
    **params: Parameters for the pipeline.

    Returns:
    float: Precision score of the bounce detection.
    """

    # Run the pipeline with given parameters
    processed_df = pipeline(df, **params)

    # Compare detected bounces with ground truth samples
    matched_samples = compare_with_sample(processed_df, tolerance=tolerance)

    # Compute precision
    precision = compute_precision(matched_samples)

    return precision
