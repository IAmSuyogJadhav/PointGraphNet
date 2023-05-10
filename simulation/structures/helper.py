import numpy as np
import pandas as pd
import os
from .Bezier import Bezier

# See if Cupy is installed
try:
    import cupy as cp

    gpu_available = True
except ImportError:  # CuPy not installed
    cp = None
    gpu_available = False


## Common shapes and structures
def trim_curve_to_length(points, min_length, max_length):
    """Trim the curve to the given length.

    Args:
        points (np.array): Points on the curve
        length (float): Length of the curve

    Returns:
        np.array: Points on the curve, trimmed to the given length
    """
    # Calculate the cumulative sum of the distances between the points, length[i] is the length of the curve up to point i
    lengths = np.cumsum(np.sqrt(np.sum((points[:-1] - points[1:]) ** 2, axis=1)))
    lengths = np.insert(
        lengths, 0, 0
    )  # insert a 0 at the beginning, to make the indexing work

    # Check if the curve is long enough
    if lengths[-1] < min_length:
        return None, lengths[-1]  # The curve is too short, can't do anything with it
    elif lengths[-1] > max_length:
        # The curve is too long, trim it to max_length
        return points[lengths <= max_length], max_length
    else:
        # The curve is just right
        return points, lengths[-1]


def _get_bezier(
    n_control_points, npoints, max_xy, zlow, zhigh, min_length, max_length, **kwargs
):
    """
    Get the bezier curve for a given set of control points.

    Args:
        n_control_points (int): Number of control points
        npoints (int): Number of points on the bezier curve
        max_xy (int): Maximum x and y coordinates, +-
        zlow (int): Minimum z coordinate
        zhigh (int): Maximum z coordinate
        min_length (int): Minimum length of the bezier curve
        max_length (int): Maximum length of the bezier curve

    Returns:
        np.array: x,y,z points of the bezier curve
    """
    # Generate the control points
    con_points = np.random.rand(n_control_points, 3) * np.array(
        [2 * max_xy, 2 * max_xy, zhigh - zlow]
    ) + np.array(
        [-max_xy, -max_xy, zlow]
    )  # Scale and shift the points to the given range

    # Get the bezier curve
    t_values = np.linspace(0, 1, npoints)
    points = Bezier.Curve(t_values, con_points)

    # Trim the curve to the given length
    points, length = trim_curve_to_length(points, min_length, max_length)
    if points is None:
        print(
            f"Curve length: {length:.2f} nm, {n_control_points} control points (too short), skipping..."
        )
        return None, None
    else:
        # print(f"Curve length: {length:.2f} nm, {n_control_points} control points")
        return points, length


def _get_sphere(
    center: tuple,
    r: float,
    points: np.ndarray,
    zlow: int,
    zhigh: int,
    max_xy: int = 2500,
    use_gpu: bool = True,
    with_normals: bool = False,
) -> np.ndarray:
    """Generate points on a sphere surface.

    Args:
        center (tuple): Center coordinates (x, y, z).
        r (float): Radius of the sphere.
        points (np.ndarray): Number of points to sample.
        zlow (int): Lower bound of the z coordinate.
        zhigh (int): Higher bound of the z coordinate.
        max_xy (int, optional): Maximum value of the x/y coordinate. The range
            is assumed to be [-max_xy, +max_xy]. Defaults to 2500.
        use_gpu (bool, optional): Whether to use GPU. Defaults to True.
        with_normals (bool, optional): Whether to return the normals. Defaults to False.

    Returns:
        np.ndarray: The sampled points.
    """
    # Check if GPU is available
    lib = cp if gpu_available and use_gpu else np
    if use_gpu and not gpu_available:
        print("[LOG] GPU not available. Using CPU instead.")

    # Choose random theta and phi
    t = lib.random.choice(
        lib.linspace(0, 360, points * 4), points
    )  # Theta (angle from z-axis)
    p = lib.random.choice(
        lib.linspace(0, 360, points * 4), points
    )  # Phi (angle from x-axis)
    center = lib.array(np.full((points, 3), center))  # For vectorization
    r = lib.full(points, r)  # For vectorization

    # Calculate x, y, z coordinates
    xyz = lib.empty((points, 3), lib.float32)
    xyz[:, 0] = center[:, 0] + r * lib.sin(lib.deg2rad(t)) * lib.cos(lib.deg2rad(p))
    xyz[:, 1] = center[:, 1] + r * lib.sin(lib.deg2rad(t)) * lib.sin(lib.deg2rad(p))
    xyz[:, 2] = center[:, 2] + r * lib.cos(lib.deg2rad(t))

    # Calculate the unit normals nx, ny, nz (unit normals), and store theta, phi (if needed)
    if with_normals:
        nxyztp = lib.empty((points, 5), lib.float32)

        nxyztp[:, 0] = lib.sin(lib.deg2rad(t)) * lib.cos(lib.deg2rad(p))
        nxyztp[:, 1] = lib.sin(lib.deg2rad(t)) * lib.sin(lib.deg2rad(p))
        nxyztp[:, 2] = lib.cos(lib.deg2rad(t))
        nxyztp[:, 3] = lib.deg2rad(t)
        nxyztp[:, 4] = lib.deg2rad(p)

    # Only keep the points that are within the specified boundary
    limit_low = lib.array([-max_xy, -max_xy, zlow])
    limit_high = lib.array([max_xy, max_xy, zhigh])
    valid = lib.all(xyz > limit_low, axis=1) & lib.all(xyz < limit_high, axis=1)

    # Concatenate the normals and return
    if with_normals:
        points = lib.concatenate((xyz[valid], nxyztp[valid]), axis=1)
        return cp.asnumpy(points) if use_gpu else points
    else:
        return cp.asnumpy(xyz[valid]) if use_gpu else xyz[valid]


## I/O functions
def save_csv(
    filepath: str, df: pd.DataFrame, header_row: bool = True, overwrite: bool = False
):
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
        df.to_csv(filepath, index=False)
    else:
        df.to_csv(filepath, index=False, header=None)


def save_parquet(filepath: str, df: pd.DataFrame, overwrite: bool = False):
    """Save the simulated points as a parquet file. Parquet does not support header rows.

    Args:
        filepath (str): The path to save the parquet file.
        df (pd.DataFrame): The simulated points.
        overwrite (bool, optional): Whether to overwrite the file if it already exists. Defaults to False.
    """

    if os.path.exists(filepath):
        print(f"{filepath} already exists.", end=" ")

        if overwrite:
            print(f"Overwriting...")
        else:
            print(f"Skipping...")
            return

    df.to_parquet(filepath, index=False)
