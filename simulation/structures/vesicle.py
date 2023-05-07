import numpy as np
import pandas as pd
from .helper import _get_sphere
from tqdm.auto import tqdm


def get_vesicle(params: dict, disable_tqdm: bool = False, with_normals: bool=False) -> pd.DataFrame:
    """Simulate vesicles with the given parameters.

    Args:
        params (dict): Dictionary of parameters. See an exmaple list of parameters in config.py.
        disable_tqdm (bool, optional): Whether to disable tqdm. Defaults to False.
        with_normals (bool, optional): Whether to return the normals. Defaults to False.

    Returns:
        pd.DataFrame: The simulated vesicles as nx4/nx8 set of points, the last column has instance ids.
    """
    rhigh = params["rhigh"]
    rlow = params["rlow"]
    zlow = params["zlow"]
    zhigh = params["zhigh"]
    max_xy = params["max_xy"]
    density = params["density"]
    number_of_vesicle = np.random.randint(
        low=params["number_of_vesicle_min"], high=params["number_of_vesicle_max"]
    )

    # Sample the vesicle centers
    num = int(2 * max_xy / rlow)
    center_x = np.random.choice(
        np.linspace(-max_xy, max_xy, num), number_of_vesicle, replace=False
    )
    center_y = np.random.choice(
        np.linspace(-max_xy, max_xy, num), number_of_vesicle, replace=False
    )
    center_z = np.random.choice(
        np.linspace(zlow, zhigh, zhigh - zlow), number_of_vesicle, replace=True
    )

    # Sample the vesicles with the given density and random radii within the range
    vesicles = []
    for x, y, z in tqdm(
        zip(center_x, center_y, center_z),
        desc="Vesicle Progress",
        total=len(center_x),
        leave=False,
        disable=disable_tqdm,
    ):
        center = np.array([x, y, z])
        r = np.random.randint(rlow, rhigh)
        area = 4 * np.pi * (r**2)
        points = int(area * density)

        ves = _get_sphere(center, r, points, zlow, zhigh, max_xy=max_xy, with_normals=with_normals)
        vesicles.append(ves)

    # Concatenate the vesicles and add instance ids
    points = np.concatenate(vesicles)
    instance_ids = np.concatenate([np.full(len(v), i) for i, v in enumerate(vesicles)])
    points = np.concatenate([points, instance_ids[:, None]], axis=1)

    # Convert to pandas dataframe
    columns = ["x", "y", "z"] + \
        (["nx", "ny", "nz", "theta", "phi"] if with_normals else []) + ["instance_id"]
    points = pd.DataFrame(points, columns=columns).reset_index(drop=True)
    points['label'] = 'vesicle'  # add label column

    return points
