npc_params = {
    "n_rings_min": 500,
    "n_rings_max": 1000,
    "n_spheres": 8,
    "density": 2500e-6,  # nm^-2
    "d_ring": 120,  # outer diameter of the ring, in nm
    "max_xy": 3200,  # +-, in nm
    "zlow": -100,  # in nm
    "zhigh": 100,  # in nm
    "center_perturbation": 0.1,  # in fraction of r_sphere
    "r_perturbation": 0.1,  # in fraction of r_sphere
    "bridson_sampling": True,  # Whether to use bridson sampling
}


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

microtubules_params = {
    "npoints": 1000,  # No. of points to sample on the CURVE, not the surface (surface points are calculated using the density)
    "zlow": -750,  # nm
    "zhigh": 750,  # nm
    "max_xy": 2500,  # +-nm
    "min_width": 13.5,  # nm
    "max_width": 14.5,  # nm
    "min_length": 1000,  # nm
    "max_length": 9000,  # nm
    "density": 8000e-6,  # nm^2
    "min_number_of_obj": 1,
    "max_number_of_obj": 5,
    "control_points_lower": 3,
    "control_points_upper": 6,
    "oversample": None,  # Oversampling not needed usually
}

mitochondria_params = {
    "npoints": 1000,  # No. of points to sample on the CURVE, not the surface (surface points are calculated using the density)
    "zlow": -750,  # nm
    "zhigh": 750,  # nm
    "max_xy": 2500,  # +-nm
    "min_width": 75,  # nm
    "max_width": 120,  # nm
    "min_length": 500,  # nm
    "max_length": 5000,  # nm
    "density": 2000e-6,  # nm^2
    "min_number_of_obj": 1,
    "max_number_of_obj": 5,
    "control_points_lower": 3,
    "control_points_upper": 6,
    "oversample": None,  # Oversampling not needed usually
}

actin_params = {
    "npoints": 1000,  # No. of points to sample on the CURVE, not the surface (surface points are calculated using the density)
    "zlow": -750,  # nm
    "zhigh": 750,  # nm
    "max_xy": 2500,  # +-nm
    "min_width": 5,  # nm  # Avg. width is 7, from https://www.ncbi.nlm.nih.gov/books/NBK9908/
    "max_width": 9,  # nm
    "min_length": 100,  # nm  # From https://doi.org/10.1016%2Fj.bpj.2010.06.025
    "max_length": 4000,  # nm  # From https://doi.org/10.1016%2Fj.bpj.2010.06.025
    "density": 10000e-6,  # nm^2
    "min_number_of_obj": 1,
    "max_number_of_obj": 6,
    "control_points_lower": 3,
    "control_points_upper": 6,
    "oversample": None,  # Oversampling not needed usually
}

params = {
    "npc": npc_params,
    "vesicle": vesicle_params,
    "microtubules": microtubules_params,
    "mito": mitochondria_params,
    "actin": actin_params,
}
