import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from .Bridson_sampling import Bridson_sampling
from .helper import _get_sphere


def get_npc(params: dict, disable_tqdm: bool = False, with_normals: bool=False) -> pd.DataFrame:
    """Simulate Nuclear Pore Complexes with the given parameters.

    Args:
        params (dict): Dictionary of parameters. See an exmaple list of parameters in config.py.
        disable_tqdm (bool, optional): Whether to disable tqdm. Defaults to False.
        with_normals (bool, optional): Whether to return normals. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing the NPC coordinates, instance ids and normals.
    """
    n_rings = np.random.randint(low=params["n_rings_min"], high=params["n_rings_max"])
    n_spheres = params["n_spheres"]
    d_ring = params["d_ring"]
    d_sphere = d_ring / 4  # DEBUG: Hardcoded for now
    max_xy = params["max_xy"]
    zlow = params["zlow"]
    zhigh = params["zhigh"]
    density = params["density"]
    center_perturbation = params["center_perturbation"]
    r_perturbation = params["r_perturbation"]
    bridson_sampling = params["bridson_sampling"]

    # Sample the ring centers
    if bridson_sampling:
        # Bridson Sampling (no intersections)
        # Source: https://www.labri.fr/perso/nrougier/from-python-to-numpy/#bridson-method
        ring_centers = Bridson_sampling(max_xy * 2, max_xy * 2, d_ring, n_rings)
        ring_centers = ring_centers - max_xy
        indices = np.random.choice(np.arange(len(ring_centers)), n_rings, replace=False)
        # print(len(ring_centers), len(indices), len(np.unique(indices))) #DEBUG
        ring_centers = ring_centers[indices]
    else:
        # Random sampling (intersection is possible)
        ring_centers = np.random.uniform(-max_xy, max_xy, size=(n_rings, 2))

    # Sample the sphere centers
    sphere_centers = np.zeros((n_rings * n_spheres, 3))
    for i, ring_center in tqdm(
        enumerate(ring_centers),
        desc="NPC Progress [1/2]",
        total=len(ring_centers),
        leave=False,
        disable=disable_tqdm,
    ):
        # Sample 8 equidistant points on each ring to place the spheres on, add a random offset to the starting point
        random_offset = np.random.uniform(0, 2 * np.pi / n_spheres)
        thetas = (
            np.linspace(0, 2 * np.pi, n_spheres, endpoint=False) + random_offset
        ) % (2 * np.pi)

        # r_ring here is the radius of the ring that passes through the _centers_ of the spheres
        r_ring = d_ring / 2 - d_sphere / 2
        z_ring = np.random.uniform(zlow, zhigh)  # z coordinate of the ring
        perturbation = np.random.uniform(
            -center_perturbation, center_perturbation, size=(n_spheres, 3)
        )  # small random perturbation added to the sphere centers

        # Compute the sphere centers
        for j in range(n_spheres):
            _center = [
                ring_center[0] + r_ring * np.cos(thetas[j]),
                ring_center[1] + r_ring * np.sin(thetas[j]),
                z_ring,
            ]
            sphere_centers[i * n_spheres + j] = _center + perturbation[j] * d_sphere / 2

    # Sample the points on the sphere surface
    points = []
    points_counter = 0  # Count points for each sphere corresponding to the current ring
    ring_lens = (
        {}
    )  # Allocate points belonging to a sequential chunk of n_spheres to the same ring
    for i, sphere_center in tqdm(
        enumerate(sphere_centers),
        desc="NPC Progress [2/2]",
        total=len(sphere_centers),
        leave=False,
        disable=disable_tqdm,
    ):
        # Add a random perturbation to the sphere radius
        r = (d_sphere / 2) * (
            1 + np.random.uniform(-r_perturbation, r_perturbation)
        )  # radius of the sphere

        # Compute the number of points on the sphere surface
        area = 4 * np.pi * (r**2)
        n_points = int(area * density)

        # Sample the points
        sphere = _get_sphere(
            center=sphere_center,
            r=r,
            points=n_points,
            zlow=zlow,
            zhigh=zhigh,
            max_xy=max_xy,
            use_gpu=False,  # is faster on CPU since we have a really small number of points
            with_normals=with_normals,
        )
        points.append(sphere)

        # For assigning instance ids later
        ring_lens[i // n_spheres] = ring_lens.get(i // n_spheres, 0) + len(sphere)

    # Concatenate the points and add instance ids
    instance_ids = np.concatenate(
        [np.full(v, i) for i, v in enumerate(ring_lens.values())]
    )
    # print(len(ring_lens), n_rings) # DEBUG
    points = np.concatenate(points)
    points = np.concatenate([points, instance_ids[:, None]], axis=1)

    # Convert to pandas dataframe
    columns = ["x", "y", "z"] + \
        (["nx", "ny", "nz", "theta", "phi"] if with_normals else []) + ["instance_id"]
    points = pd.DataFrame(points, columns=columns).reset_index(drop=True)
    points['label'] = 'npc'  # add label column

    return points