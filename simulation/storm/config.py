import datetime


npc_params = {
    "sample_percent": 0.75,
    "imprecision": 20,
    "noise_percent_fg": 0.1,
    "noise_percent_bg": 0.05,
    "noise_radius_min": 25,  # Min. distance to shift a point from its original pos
    "noise_radius_max": 60,  # Max. distance to shift a point from its original pos
    "max_xy": 2500,  # +-
    "zhigh": 100,
    "zlow": -100,
}

vesicle_params = {
    "sample_percent": 0.75,
    "imprecision": 20,
    "noise_percent_fg": 0.1,
    "noise_percent_bg": 0.05,
    "noise_radius_min": 25,  # Min. distance to shift a point from its original pos
    "noise_radius_max": 60,  # Max. distance to shift a point from its original pos
    "max_xy": 2500,  # +-
    "zhigh": 750,
    "zlow": -750,
}

microtubules_params = {
    "sample_percent": 0.75,
    "imprecision": 20,
    "noise_percent_fg": 0.1,
    "noise_percent_bg": 0.05,
    "noise_radius_min": 25,  # Min. distance to shift a point from its original pos
    "noise_radius_max": 60,  # Max. distance to shift a point from its original pos
    "max_xy": 2500,  # +-
    "zhigh": 750,
    "zlow": -750,
}

mito_params = {
    "sample_percent": 0.75,
    "imprecision": 20,
    "noise_percent_fg": 0.1,
    "noise_percent_bg": 0.05,
    "noise_radius_min": 25,  # Min. distance to shift a point from its original pos
    "noise_radius_max": 60,  # Max. distance to shift a point from its original pos
    "max_xy": 2500,  # +-
    "zhigh": 750,
    "zlow": -750,
}

actin_params = {
    "sample_percent": 0.75,
    "imprecision": 20,
    "noise_percent_fg": 0.1,
    "noise_percent_bg": 0.05,
    "noise_radius_min": 25,  # Min. distance to shift a point from its original pos
    "noise_radius_max": 60,  # Max. distance to shift a point from its original pos
    "max_xy": 2500,  # +-
    "zhigh": 750,
    "zlow": -750,
}

params = {
    "npc": npc_params,
    "vesicle": vesicle_params,
    "mito": mito_params,
    "microtubules": microtubules_params,
    "actin": actin_params,
}
