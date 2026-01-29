# Ball Bounce Detector

A library to detect padel ball direction changes (bounces) from 2D pixel coordinates.

## Installation

You can install this library directly from the git repository:

```bash
pip install git+https://your-git-repo-url.git
```

## Usage

The main function is `detect_bounces`, available directly in the `bounce_detector` package.

```python
import pandas as pd
import bounce_detector

# Load your dataframe with ball coordinates
# The dataframe must have columns for x and y (pixels).
df = pd.read_csv("your_dataset.csv")

# Define column mapping if they differ from defaults
columns_mapping = {
    "ball_x_center_col": "ball_center_x",
    "ball_y_center_col": "ball_center_y",
    # Other optional columns...
}

# Run detection
# Returns original dataframe enriched with velocity, acceleration, and detection columns
df_result = bounce_detector.detect_bounces(df, columns_mapping)

# Detected bounces are marked with 1 in 'is_bounce_detected' column
bounces = df_result[df_result['is_bounce_detected'] == 1]
print(bounces)
```

## Algorithm Details

The algorithm processes the 2D ball trajectory by calculating:
1. Coordinate smoothing.
2. Velocity and acceleration (both in XY components and polar coordinates).
3. "Non-linear" distance factors to discard false positives.
4. Acceleration peak detection indicating sudden direction changes.
5. Final classification using an ML model (Random Forest) to validate candidates.

## Project Structure

- `bounce_detector/`: Main package.
    - `pipeline/`: Mathematical processing modules.
    - `assets/`: Trained models (.pkl).
- `dev_scripts/`: Training and parameter optimization scripts.
- `datasets/`: Sample data (not included in package installation).

## Requirements

See `requirements.txt`. Mainly:
- pandas
- numpy
- scikit-learn
- joblib
