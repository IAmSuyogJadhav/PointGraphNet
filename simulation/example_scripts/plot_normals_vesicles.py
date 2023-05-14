# %%
import sys

sys.path.append("..")
from structures.vesicle import get_vesicle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib widget

# %%
vesicle_params = {
    "npoints": 1000,  # No. of points to sample on the CURVE, not the surface (surface points are calculated using the density)
    "rhigh": 500,  # nm
    "rlow": 25,  # nm
    "zlow": -750,  # nm
    "zhigh": 750,  # nm
    "max_xy": 2500,  # +-nm
    "density": 500e-6,  # nm^2
    "number_of_vesicle_min": 15,
    "number_of_vesicle_max": 35,
}

points = get_vesicle(vesicle_params, with_normals=True)
max_xy = vesicle_params["max_xy"]
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
step = 3
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
