import numpy as np
import pandas as pd
import math
from .helper import _get_sphere, _get_bezier

try:
    import cupy as cp
except ImportError:
    cp = None


def _get_cross_section_normals(points: np.ndarray):
    """ Calculate the wavefront normals at each of the given points for a noodle.
    return angles with each of the axis

    Args:
        points (np.array): Points on the curve
    
    Returns:
        np.array: angles with each of the axis, radians

    """
    # Calculate the cross section normals, by taking the difference between each point and the next
    normals = points[1:] - points[:-1]
    normals = np.insert(normals, -1, normals[-1], axis=0)  # insert the last normal
    normals[0] = - normals[0]  # flip the first normal to point towards the end of the curve
    normals /= np.linalg.norm(normals, axis=1)[:, None]  # make unit normals

    return normals

    # Calculate the angles between the normals and each of the axis
    # normal_angles =np.arccos(normals)

    # return normal_angles


def _get_noodle_surface(curve, length, width, density, with_normals, zlow, zhigh, max_xy, oversample=None, debug=False, **kwargs):
    """ Get a surface from a bezier curve.

    Args:
        curve (np.array): x,y,z points of the bezier curve
        length (float): Length of the curve
        width (float): Width of the surface
        density (float): Density of the surface
        with_normals (bool): Whether to return the normals to the surface
        zlow (int): Lower bound of the z coordinate
        zhigh (int): Higher bound of the z coordinate
        max_xy (int): Maximum value of the x/y coordinate. The range is assumed to be [-max_xy, +max_xy]
        oversample (int, optional): Factor (multiplicative) to oversample the points by. 
            Helps in simulating structures with low density, or small dimensions, or both.
            The density will be multiplied by this factor to generate a higher number of points,
            before reducing the number of points down to the desired density. Defaults to None.

    Returns:
        np.array: x,y,z points of the surface
    """
    # Calculate number of points needed to get the desired density
    area_body = np.pi * width * length  # Surface area of the cylinder body
    area_caps =  4 * np.pi * (width / 2) ** 2  # Surface area of the two hemispherical caps
    
    points_body = math.ceil(area_body * density)  # Number of points on the cylinder body (total)
    
    points_per_circle = math.ceil(points_body / len(curve))  # Number of points on the cylinder body (per circle)
    points_caps = math.ceil(area_caps * density)  # Number of points on the hemispherical caps
    points_per_cap = math.ceil(points_caps / 2)  # Number of points on each hemispherical cap

    npoints_orig = points_body + points_caps  # Total number of points, it is made sure we don't exceed this

    # Oversample the curve if needed
    if oversample is not None:
        assert isinstance(oversample, float) or isinstance(oversample, int), "Oversample must be a float or an int"

        # Oversample
        points_body = math.ceil(points_body * oversample)
        points_caps = math.ceil(points_caps * oversample)
        
        # Recalculate the number of points per circle and per cap
        points_per_circle = math.ceil(points_body / len(curve))
        points_per_cap = math.ceil(points_caps / 2)

        # print(f"Oversampling by {oversample}x: {npoints_orig} --> {points_body + points_caps} points.")  #DEBUG

    # print(f'Areas: {area_body} body, {area_caps} caps')  #DEBUG
    # print(f"Generating {points_body} points on the cylinder body, {points_caps} points on the caps")  #DEBUG
    # print(f"Points per circle: {points_per_circle}, points per cap: {points_per_cap}")  #DEBUG

    # At each curve point, get a circle of points, oriented correctly in the 3D space

    ## 1. Get cross section normals at each point on the curve
    normals = _get_cross_section_normals(curve)

    ## 2. Find a vector lying in the cross section plane at each point
    # Try one possible orthonormal vector, non-trivial soln only if normal x != 0
    orth0 = np.hstack([np.zeros((len(normals), 1)), -normals[:, 2].reshape(-1, 1), normals[:, 1].reshape(-1, 1)])
    
    # Backup vector 1, try putting y==0, non-trivial soln only if normal y != 0
    orth1 = np.hstack([normals[:, 2].reshape(-1, 1), np.zeros((len(normals), 1)), -normals[:, 0].reshape(-1, 1)])

    # Backup vector 2, try putting z==0, non-trivial soln only if normal z != 0
    orth2 = np.hstack([-normals[:, 1].reshape(-1, 1), normals[:, 0].reshape(-1, 1), np.zeros((len(normals), 1))])

    # Grab valid (non-zero) orthonormal vectors only
    orth0 = np.where(  # If the norm of the vector is 0, replace it with backup vector 1
        np.linalg.norm(orth0, axis=1).reshape(-1, 1) > 0,
        orth0,
        orth1
    )

    orth0 = np.where( # If the norm of the vector is still 0, replace it with backup vector 2
        np.linalg.norm(orth0, axis=1).reshape(-1, 1) > 0,
        orth0,
        orth2
    )

    # We need 2 planar and orthonormal unit vectors (also called orthonormal basis, apparently)
    # to define the cross section plane. We have one, calculate the other. 
    ## 3. The second vector is the cross product of the normal and the first vector!
    orth0 = orth0 / np.linalg.norm(orth0, axis=1).reshape(-1, 1)  # Normalize to get the unit vector
    orth1 = np.cross(normals, orth0) # Normalize to get the unit vector

    # Draw circles with the correct cross section orientations, by using the orthonormal basis
    # to transform the circles to the cross section plane from the XY plane
    r = width / 2 # Radius of the cross section
    circle_points = []
    for i, point in enumerate(curve): # For each point on the curve
        t = np.random.uniform(0, 2 * np.pi, points_per_circle)[..., None]  # Random thetas; N, 1

        # P = C + r * (O_0 * cos(t) + O_1 * sin(t));  P, C, O_0, O_1 are vectors.
        # P, C = Position vectors for a point on the circle and the center of the circle respectively.
        # O_0, O_1 = The orthonormal basis vectors (unit vectors) 
        xs = point[0] + r * (orth0[i][0] * np.cos(t) + orth1[i][0] * np.sin(t))  # N, 1
        ys = point[1] + r * (orth0[i][1] * np.cos(t) + orth1[i][1] * np.sin(t))  # N, 1
        zs = point[2] + r * (orth0[i][2] * np.cos(t) + orth1[i][2] * np.sin(t))  # N, 1

        if with_normals:  # If we want to return the normals, calculate them
            xyz = np.hstack([xs, ys, zs])  # N, 3
            nxyz = xyz - point  # Normal vector is the vector pointing from the center to the point on the circle
            nxyz = nxyz / np.linalg.norm(nxyz, axis=1).reshape(-1, 1)  # Normalize
            phi = np.arctan2(nxyz[:, 1], nxyz[:, 0]).reshape(-1, 1)  # Calculate phi (angle from x-axis)
            theta = np.arccos(nxyz[:, 2]).reshape(-1, 1)  # Calculate theta (angle from z-axis)

            circle_points.append(np.hstack([xyz, nxyz, phi, theta]))
        else:
            circle_points.append(np.hstack([xs, ys, zs]))

    circle_points = np.vstack(circle_points) # Stack all the points together. N, 3 or N, 8

    # Now, add hemispheres to both ends to close the surface
    # Get a complete sphere first; oversampling needed to ensure we get enough points on the caps
    CAP_OVERSAMPLING = 20
    sphere_points = [_get_sphere(
        point, r, CAP_OVERSAMPLING * points_per_cap, zlow, zhigh, max_xy,
        with_normals=with_normals, use_gpu=points_per_cap>50  # Use GPU only if there are a lot of points
        ) for point in [curve[0], curve[-1]]]
    
    # Cut the sphere according to the normal vector
    hemisphere_points = []
    for normal, sphere_ps, center in zip([normals[0], normals[-1]], sphere_points, [curve[0], curve[-1]]):
        # If the dot product of the vector from the center to the point and the normal vector is positive, keep the point, 
        # as it is on the side of the sphere that we want to keep
        keep_points = [p for p in sphere_ps if np.dot(p[:3] - center, normal) > 0]
        
        # Normals are already calculated in _get_sphere, so we don't need to calculate them again
        hemisphere_points.append(np.array(keep_points)[:points_per_cap])  # Keep only the desired number of points
    
    # Remove empty lists (if any)
    hemisphere_points = np.vstack(hemisphere_points)   # Stack all the points together. N, 3 or N, 8

    # Return the complete surface, make sure there are some points to return
    to_stack = []
    if circle_points != []:
        to_stack.append(circle_points)
    if hemisphere_points != []:
        to_stack.append(hemisphere_points)

    if len(to_stack) == 0:
        raise ValueError("No points generated. Check the parameters or try the \"oversample\" parameter.")
    elif len(to_stack) == 1:
        points = to_stack[0]
    else:
        points = np.vstack(to_stack)

    # Remove points outside the limits
    limit_low = np.array([-max_xy, -max_xy, zlow])
    limit_high = np.array([max_xy, max_xy, zhigh])
    valid = np.all(points[..., :3] > limit_low, axis=1) & np.all(points[..., :3] < limit_high, axis=1)

    points = points[valid]

    # Remove points if there are too many, otherwise do nothing
    if len(points) > npoints_orig:
        chosen = np.random.choice(points.shape[0], size=npoints_orig, replace=False)
        points = points[chosen]
        # print(f"Oversampling removed {len(points) - npoints_orig} points.")  #DEBUG

    if not debug:
        return points
    else:
        return points, curve, normals, orth0, orth1


def get_noodle(
    structure,
    npoints, zlow, zhigh, max_xy, min_width, max_width, min_length, max_length, density,
    min_number_of_obj, max_number_of_obj, control_points_lower, control_points_upper,
    with_normals=False, oversample=None):
    """
    Generate a number of random noodles with given parameters.
    Noodle == actin, microtubles or mitochondria.
    
    Args:
        structure: name of the structure
        npoints: number of points to sapmple on the bezier curve
        zlow: lower bound for z
        zhigh: upper bound for z
        max_xy: max value for x and y
        min_width: minimum width of a noodle
        max_width: maximum width of a noodle
        min_length: minimum length of a noodle
        max_length: maximum length of a noodle
        density: density of the noodles
        min_number_of_obj: minimum number of noodles
        max_number_of_obj: maximum number of noodles
        control_points_lower: minimum number of control points
        control_points_upper: maximum number of control points
        with_normals: whether to return normals or not. Default: False
        oversample (int, optional): Factor (multiplicative) to oversample the points by. 
            Helps in simulating structures with low density, or small dimensions, or both.
            The density will be multiplied by this factor to generate a higher number of points,
            before reducing the number of points down to the desired density. Defaults to None.


    Returns:
        np.ndarray: array of noodles with Nx3 or Nx8 shape (with normals)
    """
    # Sanity checks
    assert min_number_of_obj <= max_number_of_obj, "min_number_of_obj must be <= max_number_of_obj"
    assert control_points_lower <= control_points_upper, "control_points_lower must be <= control_points_upper"
    assert min_width <= max_width, "min_width must be <= max_width"
    assert min_length <= max_length, "min_length must be <= max_length"
    assert zlow <= zhigh, "zlow must be <= zhigh"
    assert max_xy > 0, "max_xy must be > 0"
    assert structure in ["actin", "microtubules", "mito"], "structure must be one of actin, microtubules or mito"

    # number of noodles
    number_of_obj = np.random.randint(min_number_of_obj, max_number_of_obj + 1)

    # number of control points
    control_points = [np.random.randint(control_points_lower, control_points_upper + 1) for _ in range(number_of_obj)]

    # Width of the noodles
    widths = [np.random.uniform(min_width, max_width) for _ in range(number_of_obj)]
    
    # Get the curve
    curves = [_get_bezier(
        n_control_points=n_control_points,
        npoints=npoints,
        max_xy=max_xy,
        zlow=zlow,
        zhigh=zhigh,
        min_length=min_length,
        max_length=max_length
    ) for n_control_points in control_points]

    # Check if the curves are valid and retry with different parameters if not
    for i, curve in enumerate(curves):
        if curve[0] is None:
            j = 0
            while curves[i][0] is None:  # Retry until a valid curve is generated
                j += 1
                print(f"Some simulations failed. Retrying with different parameters...[{j}/inf]", end="\r")
                curves[i] = _get_bezier(
                    n_control_points=np.random.randint(control_points_lower, control_points_upper + 1),
                    npoints=npoints,
                    max_xy=max_xy,
                    zlow=zlow,
                    zhigh=zhigh,
                    min_length=min_length,
                    max_length=max_length
                )


    # Get the surface points for each curve
    instances = [
        pd.DataFrame(_get_noodle_surface(
            curve=curve[0],
            length=curve[1],
            width=width,
            density=density,
            with_normals=with_normals,
            zlow=zlow,
            zhigh=zhigh,
            max_xy=max_xy,
            oversample=oversample
        )) for curve, width in zip(curves, widths)
    ]

    # Assign instance ids
    for i, instance in enumerate(instances):
        instance.columns = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'theta', 'phi'] if with_normals else ['x', 'y', 'z']
        instance['instance_id'] = i
        instance['label'] = structure
        
    # Concatenate all the points
    points = pd.concat(instances, ignore_index=True).reset_index(drop=True)

    return points


def get_microtubules(params, **kwargs):
    """
    Generate a number of random microtubules with given parameters.
    See simulate_noodle for more details.

    Args:
        params: parameters for the simulation
    """
    return get_noodle(structure="microtubules", **params, **kwargs)


def get_actin(params, **kwargs):
    """
    Generate a number of random actin filaments with given parameters.
    See simulate_noodle for more details.

    Args:
        params: parameters for the simulation
    """
    return get_noodle(structure="actin", **params, **kwargs)


def get_mito(params, **kwargs):
    """
    Generate a number of random mitochondria with given parameters.
    See simulate_noodle for more details.

    Args:
        params: parameters for the simulation
    """
    return get_noodle(structure="mito", **params, **kwargs)
