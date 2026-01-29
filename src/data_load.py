import os
import pandas as pd


def load_dataset_from_directory(directory_path, use_sample=True):
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The directory {directory_path} does not exist.")
    dataset_name = os.path.basename(directory_path)
    dataset_file = os.path.join(directory_path, f"{dataset_name}.csv")
    if not os.path.isfile(dataset_file):
        raise FileNotFoundError(f"The dataset file {dataset_file} does not exist in the directory.")
    
    sample_file = os.path.join(directory_path, f"{dataset_name}-bounces-sample.csv")
    if not os.path.isfile(sample_file):
        raise FileNotFoundError(f"The sample file {sample_file} does not exist in the directory.")

    df = pd.read_csv(dataset_file)
    df['frame'] = df.index
    
    if not use_sample:
        return df

    sample_df = pd.read_csv(sample_file)

    df = df.iloc[:sample_df['bounce_frame'].max()+1].copy()
    df['is_bounce'] = False

    bounce_frames = set(sample_df['bounce_frame'])
    df.loc[df['frame'].isin(bounce_frames), 'is_bounce'] = True

    return df

