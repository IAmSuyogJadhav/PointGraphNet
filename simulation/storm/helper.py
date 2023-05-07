import pandas as pd
import numpy as np


def add_imprecision(points: pd.DataFrame, shift_radius: float = 20) -> pd.DataFrame:
    """Adds imprecision to the points, caused due to the length of the dye molecules. 
    By default, adds upto +- 20 nanometers to each point in 3D space.

    Args:
        points (pd.DataFrame): The points to add imprecision to.
        shift_radius (float, optional): The maximum shift in nanometers. Defaults to 20.

    Returns:
        pd.DataFrame: The points with imprecision added.
    """
    points_shifted = points.copy()
    points_shifted[['x', 'y', 'z']] += np.random.uniform(-shift_radius, shift_radius, size=(points.shape[0], 3))
    return points_shifted


def sample_points(points: pd.DataFrame, sample_percent: float) -> pd.DataFrame:
    """Samples a percentage of the points.

    Args:
        points (pd.DataFrame): The points to sample.
        sample_percent (float): The percentage of points to keep.

    Returns:
        pd.DataFrame: The sampled points.
    """
    return points.sample(frac=sample_percent, replace=False)


def random_uniform_discontinuous(a, b, c, d, size=1):
    """Generates a random number from a uniform distribution within the discontinuous range [a, b] U [c, d],
    where a < b < c < d. Basically np.random.uniform, but with a discontinuity.

    Args:
        a (float): The lower bound of the first range.
        b (float): The upper bound of the first range.
        c (float): The lower bound of the second range.
        d (float): The upper bound of the second range.
        size (int, optional): The number of random numbers to generate. Defaults to 1.

    Returns:
        float: The generated random number within the two bounds.

    Source:
        https://stackoverflow.com/a/56230402
    """
    displacement = np.random.uniform(a - b, d - c, size=size)
    displacement += np.where(displacement < 0, b, c)
    # print(str(displacement.min())) #DEBUG
    return displacement


def get_foreground_noise(points: pd.DataFrame, noise_percent: float, noise_radius_min: int=20, noise_radius_max: int=60) -> pd.DataFrame:
    """Generates and returns random noisy points to simulate mis-localisation by thunderstorm of the actual points.

    Args:
        points (pd.DataFrame): The points to add noise to.
        noise_percent (float): The percentage of noisy points as a fraction 
            of the total number of points.
        noise_radius_min (int, optional): The minimum distance of the noisy point 
            from the original point's position. Defaults to 20.
        noise_radius_max (int, optional): The maximum distance of the noisy point
            from the original point's position. Defaults to 60.

    Returns:
        pd.DataFrame: The noisy points.
    """
    points_noisy = points.copy()
    points_noisy = points_noisy.sample(frac=noise_percent, replace=True)  # Sample with replacement, to allow for multiple noisy points per original point
    points_noisy['label'] = 'noise_fg'

    # Sample displacement for each point from the range [-noise_radius_max, -noise_radius_min] U [noise_radius_min, noise_radius_max]
    points_noisy[['x', 'y', 'z']] += random_uniform_discontinuous(-noise_radius_max, -noise_radius_min, noise_radius_min, noise_radius_max, size=(points_noisy.shape[0], 3))
    # points = pd.concat([points, points_noisy])
    # return points
    return points_noisy


def get_background_noise(points: pd.DataFrame, noise_percent: float, max_xy: float, zlow: float, zhigh: float) -> pd.DataFrame:
    """Generates and returns random points to be added to the background as noise with given noise_percent.

    Args:
        points (pd.DataFrame): The points to add noise to.
        noise_percent (float): The percentage of noisy points as a fraction 
            of the total number of points.
        max_xy (float): The maximum x and y coordinates for the background noise (+-)
        zlow (float): The minimum z coordinate for the background noise
        zhigh (float): The maximum z coordinate for the background noise

    Returns:
        pd.DataFrame: The noisy points.
    """
    points_noisy = points.copy()
    points_noisy = points_noisy.sample(frac=noise_percent)

    # points_noisy['x'] = np.random.uniform(points_noisy['x'].min(), points_noisy['x'].max(), size=points_noisy.shape[0])
    # points_noisy['y'] = np.random.uniform(points_noisy['y'].min(), points_noisy['y'].max(), size=points_noisy.shape[0])
    # points_noisy['z'] = np.random.uniform(points_noisy['z'].min(), points_noisy['z'].max(), size=points_noisy.shape[0])
    points_noisy['x'] = np.random.uniform(-max_xy, max_xy, size=points_noisy.shape[0])
    points_noisy['y'] = np.random.uniform(-max_xy, max_xy, size=points_noisy.shape[0])
    points_noisy['z'] = np.random.uniform(zlow, zhigh, size=points_noisy.shape[0])
    points_noisy['label'] = 'noise_bg'
    points_noisy['instance_id'] = -1
    
    # Make sure normals are set to invalid values
    if 'nx' in points_noisy.columns:
        points_noisy['nx'] = 0
        points_noisy['ny'] = 0
        points_noisy['nz'] = 0
        points_noisy['theta'] = 0
        points_noisy['phi'] = 0
    
    # Check if ground truth is present for the points and set that to same values as background noise
    if 'x_gt' in points_noisy.columns:
        points_noisy['x_gt'] = points_noisy['x']
        points_noisy['y_gt'] = points_noisy['y']
        points_noisy['z_gt'] = points_noisy['z']
    # points = pd.concat([points, points_noisy])
    # return points
    return points_noisy


def simulate_thunderstorm(
    points: pd.DataFrame, sample_percent: float=0.75, noise_percent_fg: float=0.15, 
    imprecision: float=20, noise_radius_min: int=20, noise_radius_max: int=60, 
    noise_percent_bg: float=0.05, max_xy: float=2500, zlow: float=-750, zhigh: float=750
    ) -> pd.DataFrame:
    """Simulates thunderstorm by sampling points, adding noise, and adding localisation imprecision.

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
        max_xy (float, optional): The maximum x and y coordinates for the background noise (+-). Defaults to 2500.
        zlow (float, optional): The minimum z coordinate for the background noise. Defaults to -750.
        zhigh (float, optional): The maximum z coordinate for the background noise. Defaults to 750.

    Returns:
        pd.DataFrame: The simulated thunderstorm points.
    """
    # Step 1: Some of the dye molecules are not detected by thunderstorm, so we sample a percentage of the points
    points = sample_points(points=points, sample_percent=sample_percent)

    # Step 2: The dye molecules themseles have a small imprecision due to their own length
    points = add_imprecision(points=points, shift_radius=imprecision)

    # Step 3: Some of the molecules are mis-localised by thunderstorm, so we add noise to some of the points
    noise_fg = get_foreground_noise(points=points, noise_percent=noise_percent_fg, noise_radius_min=noise_radius_min, noise_radius_max=noise_radius_max)

    # Step 4: Add random noise to the background as an additional source of noise
    noise_bg = get_background_noise(points=points, noise_percent=noise_percent_bg, max_xy=max_xy, zlow=zlow, zhigh=zhigh)

    points = pd.concat([points, noise_fg, noise_bg], ignore_index=True).reset_index(drop=True)
    return points