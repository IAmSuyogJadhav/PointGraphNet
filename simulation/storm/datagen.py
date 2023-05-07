from multiprocessing import Pool
from typing import Generator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from config import params
from tqdm.auto import tqdm
import traceback
import os, glob
import json
from helper import simulate_thunderstorm


def load_csv(filename: str, label: str=None) -> pd.DataFrame:
    """ Load a csv file into a pandas DataFrame.

    Args:
        filename (str): The name of the file to load.
        label (str): The label to assign to the data. Deprecated.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    df = pd.read_csv(filename)
    # if len(df.columns) == 4:
    #     df.columns = ['x', 'y', 'z', 'instance_id']
    # elif len(df.columns) == 5:
    #     df.columns = ['x', 'y', 'z', 'instance_id', 'label']
    # elif len(df.columns) == 10:
    #     df.columns = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'theta', 'phi', 'instance_id', 'label']
    # else:
    #     raise ValueError(f"Unexpected number of columns in {filename}: {len(df.columns)}")

    return df


def load_parquet(filename: str, label: str=None) -> pd.DataFrame:
    """ Load a parquet file into a pandas DataFrame.

    Args:
        filename (str): The name of the file to load.
        label (str): The label to assign to the data. Deprecated.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    df = pd.read_parquet(filename)
    # if len(df.columns) == 4:
    #     df.columns = ['x', 'y', 'z', 'instance_id']
    # elif len(df.columns) == 5:
    #     df.columns = ['x', 'y', 'z', 'instance_id', 'label']
    # elif len(df.columns) == 10:
    #     df.columns = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'theta', 'phi', 'instance_id', 'label']
    # else:
    #     raise ValueError(f"Unexpected number of columns in {filename}: {len(df.columns)}")

    return df


def simulate_and_save(points: pd.DataFrame, filename: str, sample_percent: float=0.75, noise_percent_fg: float=0.15, imprecision: float=20, noise_radius_min: int=20, noise_radius_max: int=60, noise_percent_bg: float=0.05, overwrite: bool=False, format='parquet'):
    """Simulates thunderstorm by sampling points, adding noise, and adding localisation imprecision, and saves the simulated points to a CSV file.

    Args:
        points (pd.DataFrame): The points to simulate thunderstorm on.
        sample_percent (float, optional): The percentage of points to sample. Defaults to 0.75.
        noise_percent_fg (float, optional): The percentage of noisy points as a fraction 
            of the total number of points. Defaults to 0.15.
        imprecision (float, optional): The maximum shift in nanometers. Defaults to 20.
        noise_radius_min (int, optional): The minimum distance of the noisy point 
            from the original point's position. Defaults to 20.
        noise_radius_max (int, optional): The maximum distance of the noisy point
            from the original point's position. Defaults to 60.
        noise_percent_bg (float, optional): The percentage of random noisy points in the background as a fraction
            of the total number of points. Defaults to 0.05.
        filename (str): The name of the file to save the simulated points to.
        overwrite (bool, optional): Whether to overwrite the file if it already exists. Defaults to False.
        format (str, optional): The format to save the file in. Defaults to 'parquet'. Options are 'parquet' and 'csv'.
    """
    try:
        points = simulate_thunderstorm(points, sample_percent, noise_percent_fg, imprecision, noise_radius_min, noise_radius_max, noise_percent_bg)
        if format == 'parquet':
            save_parquet(filepath=filename, data=points, overwrite=overwrite)
        elif format == 'csv':
            save_csv(filepath=filename, data=points, overwrite=overwrite)
        else:
            raise ValueError(f"Unknown format {format}")
    except Exception as e:
        return traceback.format_exc()


def show_2d(points: pd.DataFrame, title: str = 'Points (2D)', crop_idxs: tuple = (None, None, None, None), labels: bool=False):
    """Shows a 2D plot of the points.

    Args:
        points (pd.DataFrame): The points to plot.
        title (str, optional): The title of the plot. Defaults to 'Points (2D)'.
        crop_idxs (tuple, optional): The indices to crop the plot to, in the format (x_min, x_max, y_min, y_max). Defaults to (None, None, None, None).
        labels (bool, optional): Whether to denote the labels of the points via different colors. Defaults to False.
    """

    # Plot different labels in different colors
    plt.subplot(1, 2, 1); plt.title('Classes')
    if labels:
        for label in points['label'].unique():
            plt.scatter(points[points['label'] == label]['x'], points[points['label'] == label]['y'], label=label, marker='.', s=2**2)
    else:
        plt.scatter(points['x'], points['y'], marker='.', s=2**2)

    # Crop the plot
    plt.xlim(crop_idxs[0], crop_idxs[1]); plt.ylim(crop_idxs[2], crop_idxs[3])
    plt.legend()

    # Plot different instances in different colors
    plt.subplot(1, 2, 2); plt.title('Instances')

    for instance in points['instance_id'].unique():
        plt.scatter(points[points['instance_id'] == instance]['x'], points[points['instance_id'] == instance]['y'], label=instance, marker='.', s=2**2)

    # Crop the plot
    plt.xlim(crop_idxs[0], crop_idxs[1]); plt.ylim(crop_idxs[2], crop_idxs[3])

    # Set title and show
    plt.suptitle(title)
    plt.show()


def save_parquet(
    filepath: str, data: np.ndarray, label: str=None, overwrite: bool = False
):
    """Save the simulated points as a parquet file. Parquet does not support header rows.

    Args:
        filepath (str): The path to save the parquet file.
        data (np.ndarray): The simulated points as nx3 array.
        label (str): The label of the points. Not used, but included for uniformity with simulation/datagen.py code.
        overwrite (bool, optional): Whether to overwrite the file if it already exists. Defaults to False.
    """

    if os.path.exists(filepath):
        print(f"{filepath} already exists.", end=" ")

        if overwrite:
            print(f"Overwriting...")
        else:
            print(f"Skipping...")
            return

    # df = pd.DataFrame(data, columns=["x", "y", "z", "instance_id", "label"])
    df = pd.DataFrame(data)
    # if label is not None:
    #     df["label"] = label
    df.to_parquet(filepath, index=False)


def save_csv(
    filepath: str, data: pd.DataFrame, header_row: bool = True, overwrite: bool = False):
    """Save the simulated points as a csv file, for use with testSTORM.

    Args:
        filepath (str): The path to save the csv file.
        df (pd.DataFrame): The simulated points.
        header_row (bool, optional): Whether to include the header row. Defaults to False, for use with testSTORM.
        overwrite (bool, optional): Whether to overwrite the file if it already exists. Defaults to False.
    """

    if os.path.exists(filepath):
        print(f"{filepath} already exists.", end=" ")

        if overwrite:
            print(f"Overwriting...")
        else:
            print(f"Skipping...")
            return

    if header_row:
        data.to_csv(filepath, index=False)
    else:
        data.to_csv(filepath, index=False, header=None)


def rotating_cursor() -> str:
    """A generator that yields a rotating cursor."""
    while True:
        for cursor in "|/-\\":
            yield cursor


def _update_pbar(pbar: tqdm, cur: Generator, desc: str):
    """Updates the progress bar and the description using the given cursor generator.

    Args:
        pbar (tqdm): The progress bar.
        cur (Generator): The cursor generator.
        desc (str): The description to use. Cursor will be appended to this.
    """
    pbar.update(1)
    pbar.set_description(f"{desc} {next(cur)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Simulate thunderstorm on the given ideal point clouds.')
    parser.add_argument('structure', type=str, help='The structure to simulate thunderstorm on (npc, vesicle, mito, actin, microtubules).')
    parser.add_argument('--print_params', action='store_true', help='Print the default parameters (view and edit in config.py).')

    parser.add_argument('-n', type=int, default=None, help='The number of point clouds to simulate thunderstorm on (first n will be picked). Defaults to all.')
    parser.add_argument('-i', '--data_path', type=str, default=None, help='The path to the data directory.')
    parser.add_argument('--sample_percent', type=float, default=None, help='The percentage of points to keep.')
    parser.add_argument('--imprecision', type=float, default=None, help='The maximum shift in nanometers.')
    parser.add_argument('--noise_percent_fg', type=float, default=None, help='The percentage of foreground noise to add.')
    parser.add_argument('--noise_percent_bg', type=float, default=None, help='The percentage of background noise to add.')
    parser.add_argument('--noise_radius_min', type=float, default=None, help='The minimum radius of the noise.')
    parser.add_argument('--noise_radius_max', type=float, default=None, help='The maximum radius of the noise.')
    parser.add_argument('-o', '--output_path', type=str, default=None, help='The path to the output directory.')

    parser.add_argument('--format', type=str, default='parquet', help='The format to read/write the simulated point clouds in. Defaults to parquet. Supported formats: parquet, csv.')
    parser.add_argument('--seed', type=int, default=None, help='The random seed to use. If not specified, no seed is set.')
    parser.add_argument('--workers', type=int, default=8, help='The number of workers to use for multiprocessing. Defaults to 8.')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite the output files if they already exist.')
    args = parser.parse_args()

    # If the user did not specify any parameters, use the default parameters
    # params_ = eval(f"{args.structure}_params")
    params_ = params[args.structure]
    for key, value in params_.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    if args.print_params:
        print(f"Using the following parameters for {args.structure}:")
        for key in args:
            print(f"\t{key}: {getattr(args, key)}")

    # Sanity checks
    assert args.structure in params, f"Unknown structure: {args.structure}."
    assert args.format.strip('.').lower() in ["parquet", "csv"], f"Unknown format: {args.format}."
    assert args.data_path is not None, "Please specify the path to the data directory."
    assert args.output_path is not None, "Please specify the path to the output directory."
    assert args.sample_percent >= 0 and args.sample_percent <= 1, "Sample percent must be between 0 and 1."
    assert args.noise_percent_fg >= 0 and args.noise_percent_fg <= 1, "Noise percent (foreground) must be between 0 and 1."
    assert args.noise_percent_bg >= 0 and args.noise_percent_bg <= 1, "Noise percent (background) must be between 0 and 1."
    assert args.noise_radius_min >= 0, "Noise radius (min) must be non-negative."
    assert args.noise_radius_max >= 0, "Noise radius (max) must be non-negative."
    assert args.noise_radius_min <= args.noise_radius_max, "Noise radius (min) must be less than or equal to noise radius (max)."
    assert args.imprecision >= 0, "Imprecision must be non-negative."

    return args


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Load the points
    files = glob.glob(os.path.join(args.data_path, f'{args.structure}_*.{args.format.strip(".").lower()}'))
    files = sorted(files)  # To keep 1:1 correspondence with the input files
    data = [load_parquet(file) for file in files] if args.format == "parquet" else [load_csv(file) for file in files]
    assert len(data) > 0, f"No point clouds found in {args.data_path}."
    if args.n is None:
        print(f"Using all {len(data)} point clouds.")
        args.n = len(data)  # Use all point clouds

    # Set the random seed
    if args.seed is not None:
        np.random.seed(args.seed)

    # Create the output directory
    os.makedirs(args.output_path, exist_ok=True)

    if args.n == 1:
        fname = os.path.join(args.output_path, f"{args.structure}.{args.format.strip('.').lower()}")
        simulate_and_save(
            points=data[0],
            filename=fname,
            sample_percent=args.sample_percent,
            noise_percent_fg=args.noise_percent_fg,
            imprecision=args.imprecision,
            noise_radius_min=args.noise_radius_min,
            noise_radius_max=args.noise_radius_max,
            noise_percent_bg=args.noise_percent_bg,
            overwrite=args.overwrite,
        )

    else:
        print = tqdm.write  # Works better with progress bars
        print(f"Using {args.workers} workers.")
        cur = rotating_cursor()  # Small rotating cursor

        with tqdm(total=args.n, desc="Simulating thunderstorm") as pbar:
            with Pool(args.workers) as pool:
                for i in range(args.n):
                    fname = os.path.join(args.output_path, f"{args.structure}_{i}.{args.format.strip('.').lower()}")
                    ret = pool.apply_async(
                        simulate_and_save,
                        args=(
                            data[i],
                            fname,
                            args.sample_percent,
                            args.noise_percent_fg,
                            args.imprecision,
                            args.noise_radius_min,
                            args.noise_radius_max,
                            args.noise_percent_bg,
                            args.overwrite,
                        ),
                        callback=lambda _: _update_pbar(pbar, cur, "Simulating thunderstorm"),
                    )
                    
                    # Check for errors
                    if ret.get() is not None:
                        print(ret.get())
                
                pool.close()
                pool.join()

    # Save the parameters
    with open(os.path.join(args.output_path, "params.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
