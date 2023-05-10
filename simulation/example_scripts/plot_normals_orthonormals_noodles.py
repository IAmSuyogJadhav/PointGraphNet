# %%
import sys

sys.path.append("..")
from structures.noodles import *
from structures.noodles import _get_bezier, _get_noodle_surface
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib widget

# %%
mitochondria_params = {
    "npoints": 1000,  # No. of points to sample on the CURVE, not the surface (surface points are calculated using the density)
    "zlow": -750,  # nm
    "zhigh": 750,  # nm
    "max_xy": 2500,  # +-nm
    "min_width": 599,  # nm
    "max_width": 600,  # nm
    "min_length": 500,  # nm
    "max_length": 5000,  # nm
    "density": 2000e-6,  # nm^2
    "min_number_of_obj": 1,
    "max_number_of_obj": 1,
    "control_points_lower": 3,
    "control_points_upper": 6,
    "oversample": None,  # Oversampling not needed usually
}

# %%
npoints = mitochondria_params["npoints"]
zlow = mitochondria_params["zlow"]
zhigh = mitochondria_params["zhigh"]
max_xy = mitochondria_params["max_xy"]
min_width = mitochondria_params["min_width"]
max_width = mitochondria_params["max_width"]
min_length = mitochondria_params["min_length"]
max_length = mitochondria_params["max_length"]
density = mitochondria_params["density"]
min_number_of_obj = mitochondria_params["min_number_of_obj"]
max_number_of_obj = mitochondria_params["max_number_of_obj"]
control_points_lower = mitochondria_params["control_points_lower"]
control_points_upper = mitochondria_params["control_points_upper"]
oversample = mitochondria_params["oversample"]

# number of noodles
number_of_obj = np.random.randint(min_number_of_obj, max_number_of_obj + 1)

# number of control points
control_points = [
    np.random.randint(control_points_lower, control_points_upper + 1)
    for _ in range(number_of_obj)
]

# Width of the noodles
widths = [np.random.uniform(min_width, max_width) for _ in range(number_of_obj)]

curves = [
    _get_bezier(
        n_control_points=n_control_points,
        npoints=npoints,
        max_xy=max_xy,
        zlow=zlow,
        zhigh=zhigh,
        min_length=min_length,
        max_length=max_length,
    )
    for n_control_points in control_points
]

# Get the surface points
instances = [
    _get_noodle_surface(
        curve=curve[0],
        length=curve[1],
        width=width,
        density=density,
        with_normals=True,
        zlow=zlow,
        zhigh=zhigh,
        max_xy=max_xy,
        oversample=oversample,
        debug=True,
    )
    for curve, width in zip(curves, widths)
]

points, curve, normals, orth0, orth1 = instances[0]
points = pd.DataFrame(points, columns=["x", "y", "z", "nx", "ny", "nz", "theta", "phi"])

# %%
# Plot the points in 3D, marker size = 0.1
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    points["x"], points["y"], points["z"], c="black", alpha=0.8, marker=".", s=0.1
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# Plot the cross section normal vectors and corresponding orthogonal vectors
step = 25
ax.quiver(
    curve[::step, 0],
    curve[::step, 1],
    curve[::step, 2],
    normals[::step, 0],
    normals[::step, 1],
    normals[::step, 2],
    color="red",
    label="Cross-section normals",
    length=500,
)
ax.quiver(
    curve[::step, 0],
    curve[::step, 1],
    curve[::step, 2],
    orth0[::step, 0],
    orth0[::step, 1],
    orth0[::step, 2],
    color="blue",
    label="Orthogonal vectors",
    length=500,
)
ax.quiver(
    curve[::step, 0],
    curve[::step, 1],
    curve[::step, 2],
    orth1[::step, 0],
    orth1[::step, 1],
    orth1[::step, 2],
    color="green",
    label="Orthogonal vectors",
    length=500,
)
ax.legend()

ax.set_xlim(-max_xy, max_xy)
ax.set_ylim(-max_xy, max_xy)
ax.set_zlim(-max_xy, max_xy)

plt.show()

# %%
# Plot the points in 3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    points["x"],
    points["y"],
    points["z"],
    c=points["z"],
    alpha=0.5,
    marker=".",
    label="Surface points",
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# Plot the surface normal vectors
step = 5
ax.quiver(
    points["x"][::step],
    points["y"][::step],
    points["z"][::step],
    points["nx"][::step],
    points["ny"][::step],
    points["nz"][::step],
    color="blue",
    label="Surface normals",
    length=200,
    alpha=0.3,
)
ax.legend()

ax.set_xlim(-max_xy, max_xy)
ax.set_ylim(-max_xy, max_xy)
ax.set_zlim(-max_xy, max_xy)
plt.show()
