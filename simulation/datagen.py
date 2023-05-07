import os
import sys
import argparse
from typing import Generator
from multiprocessing import Pool
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import json
from structures import get_actin, get_microtubules, get_mito, get_npc, get_vesicle
from structures.helper import save_csv, save_parquet
from storm.helper import simulate_thunderstorm
import traceback
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from threading import Thread


def simulate(
    structure_name: str, params: dict, disable_tqdm: bool = False, with_normals: bool = False
) -> pd.DataFrame:
    """Simulates the given structure, subject to the given parameters.

    Args:
        structure_name (str): Name of the structure. Check datagen.py for the list of available structures.
        params (dict): The parameters' dictionary for the structure.
        disable_tqdm (bool, optional): Whether to disable tqdm progress bar. Defaults to False.
        with_normals (bool, optional): Whether to return the normals of the simulated points. Defaults to False.

    Returns:
        np.ndarray: Simulated points as nx3 array.
    """

    if structure_name == "vesicle":
        return get_vesicle(params, disable_tqdm=disable_tqdm, with_normals=with_normals)
    elif structure_name == "npc":
        return get_npc(params, disable_tqdm=disable_tqdm, with_normals=with_normals)
    elif structure_name == "microtubules":
        return get_microtubules(params, with_normals=with_normals)
    elif structure_name == "mito":
        return get_mito(params, with_normals=with_normals)
    elif structure_name == "actin":
        return get_actin(params, with_normals=with_normals)
    else:
        raise ValueError(f"Invalid structure name: {structure_name}")


def simulate_and_save(
    structure: str,
    params: dict,
    fname: str,
    ftype: str = "parquet",
    disable_tqdm: bool = False,
    header: bool = False,
    overwrite: bool = False,
    with_normals: bool = False,
    storm_params: dict = None,
    no_storm: bool = False,
    seed: int = None,
):
    """Simulates the given structure, subject to the given parameters, and saves it as a csv file.
    Also prevents unnecessary computation if the file already exists and overwrite is False, allowing for
    faster resuming from a previous run. Also catches errors and prints them to the console (useful for multithreading).

    Args:
        structure (str): Name of the structure. Check datagen.py for the list of available structures.
        params (dict): The parameters' dictionary for the structure.
        fname (str): The path to save the file.
        ftype (str, optional): The file type [csv, parquet]. Defaults to parquet.
        disable_tqdm (bool, optional): Whether to disable tqdm progress bar. Defaults to False.
        header (bool, optional): Whether to include the header row (only for csv). Defaults to False, for use with testSTORM.
        overwrite (bool, optional): Whether to overwrite the file if it already exists. Defaults to False.
        with_normals (bool, optional): Whether to include the normals in the output file. Defaults to False.
        storm_params (dict, optional): The parameters for thunderstorm. Defaults to None.
        no_storm (bool, optional): Whether to disable thunderstorm noise addition to the points. Defaults to False.
        seed (int, optional): The seed for the random number generator. Defaults to None.
    """
    # Reset the random seed (needs to be done in each process, 
    # otherwise multiple processes end up generating the same structures!)
    # See: https://stackoverflow.com/q/9209078/9168131
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()

    try:
        # Prevent unwanted simulations if the file already exists and overwrite is not set, also allows for resuming from a previous run
        points = (
            simulate(structure, params, disable_tqdm=disable_tqdm, with_normals=with_normals)
            if overwrite or not os.path.exists(fname)
            else None
        )

        # Copy the original x, y, z columns to use as gt
        if points is not None:
            points['x_gt'] = points['x']
            points['y_gt'] = points['y']
            points['z_gt'] = points['z']

        # Add thunderstorm noise to the points
        if points is not None and not no_storm:
            points = simulate_thunderstorm(points=points, **storm_params)

        if ftype == "csv":
            save_csv(filepath=fname, df=points, header_row=header, overwrite=overwrite)
        elif ftype == "parquet":
            save_parquet(filepath=fname, df=points, overwrite=overwrite)

    except Exception as e:
        traceback.print_exc()
        # print(f"Error: {e}")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Simulate a structure and save the points as a csv/parquet file."
    )
    parser.add_argument(
        "-n", type=int, default=1, help="Number of structures to simulate."
    )
    parser.add_argument(
        "structure",
        type=str,
        help="The structure to simulate. Available options: vesicle, npc, mito, actin, microtubules, all",
    )
    parser.add_argument(
        "--with-normals",
        action="store_true",
        help="Whether to include the normals in the output file.",
    )

    parser.add_argument(
        "--no-storm",
        action="store_true",
        help="Specify to NOT add thunderstorm noise to the points.",
    )

    # parser.add_argument(
    #     "params",
    #     type=str,
    #     help="The path to the parameters' json file.",
    # )

    parser.add_argument(
        "-o",
        type=str,
        default=None,
        help="The output directory prefix for the files. The files will be saved as {prefix}/{structure_name}_{i}.[csv/parquet]."
        "Defaults to 'data/{ddmmyy-hhmms}'.",
    )
    parser.add_argument(
        "-f",
        "--ftype",
        type=str,
        default="parquet",
        help="The file type to save the points as. Available file types: csv, parquet. Defaults to parquet.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of workers to use for multiprocessing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--header",
        action="store_true",
        help="Whether to include the header row in the file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="The random seed to use. Defaults to None, which does not set the seed.",
    )

    return parser.parse_args()


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


def main(
    structure: str,
    params: dict,
    prefix: str,
    ftype: str,
    n: int,
    overwrite: bool = False,
    header: bool = False,
    workers: int = 8,
    seed: int = None,
    with_normals: bool = False,
    storm_params: dict = None,
    no_storm: bool = False,
):
    """The main function for simulating a set of given structures.

    Args:
        structure (str): The structure to simulate.
        params (dict): The parameters' dictionary for the structure.
        prefix (str): The output directory prefix for the files. The files will be saved as {prefix}/{structure_name}_{i}.[csv/parquet].
        ftype (str): The file type to save the points as. Available file types: csv, parquet.
        n (int): Number of structures to simulate.
        overwrite (bool, optional): Whether to overwrite the output file if it already exists. Defaults to False.
        header (bool, optional): Whether to include the header row in the file (only for csv files). Defaults to False.
        workers (int, optional): Number of workers to use for multithreading. Defaults to 8.
        seed (int, optional): The random seed to use. Defaults to None, no seed is set.
        with_normals (bool, optional): Whether to include the normals in the output file. Defaults to False.
        storm_params (dict, optional): The parameters' dictionary for simulating thunderstorm. Defaults to None.
        no_storm (bool, optional): Whether to disable thunderstorm noise simulation. Defaults to False.
    """
    if args.n > 1:
        print = tqdm.write  # Works nicer with progress bars

    # Create the output directory
    os.makedirs(prefix, exist_ok=True)

    # Make sure the seed is truly unique for each structure (in case multiple structures are simulated with same seed)
    seed = seed + hash(structure) if seed is not None else None

    # Simulate the structure(s)
    if n == 1:  # No progress bar for a single structure
        fname = os.path.join(prefix, f"{structure}.{ftype}")
        simulate_and_save(
            structure=structure,
            params=params,
            fname=fname,
            ftype=ftype,
            overwrite=overwrite,
            disable_tqdm=True,
            header=header,
            with_normals=with_normals,
            storm_params=storm_params,
            no_storm=no_storm,
            seed=seed,
        )

    else:
        print(f"Using {workers} workers...")
        cur = rotating_cursor()  # Small rotating cursor

        # Get unique seed for each structure
        seeds = [seed + i for i in range(n)] if seed is not None else [None for _ in range(n)]
        with tqdm(total=args.n, desc=f"Progress {next(cur)}") as pbar:  # Progress bar

            # use multiprocessing pool
            with Pool(workers) as pool:
                for i in range(n):
                    fname = os.path.join(prefix, f"{structure}_{i}.{ftype}")
                    pool.apply_async(
                        simulate_and_save,
                        # args=(
                        #     structure,
                        #     params,
                        #     fname,
                        #     ftype, # File type
                        #     True,  # disable_tqdm
                        #     header,
                        #     overwrite,
                        #     with_normals,
                        # ),
                        kwds={
                            "structure": structure,
                            "params": params,
                            "fname": fname,
                            "ftype": ftype,
                            "disable_tqdm": True,
                            "header": header,
                            "overwrite": overwrite,
                            "with_normals": with_normals,
                            "storm_params": storm_params,
                            "no_storm": no_storm,
                            "seed": seeds[i],
                        },
                        callback=lambda _: _update_pbar(
                            pbar, cur, "Progress"
                        ),  # Update the progress bar
                    )
                pool.close()
                pool.join()

            # # using threading pool
            # with ThreadPoolExecutor(max_workers=workers) as executor:  # Multithreading
            #     futures = [
            #         executor.submit(
            #             simulate_and_save,
            #             structure=structure,
            #             params=params,
            #             fname=os.path.join(prefix, f"{structure}_{i}.csv"),
            #             disable_tqdm=True,
            #             header=header,
            #             overwrite=overwrite,
            #         )
            #         for i in range(n)
            #     ]

            # # Update the progress bar
            # for _ in as_completed(futures):
            #     pbar.update(1)
            #     pbar.set_description(f"Progress {next(cur)}")


if __name__ == "__main__":
    from config import params
    from storm.config import params as storm_params
    import datetime

    # Parse the command line arguments
    args = parse_args()

    # Sanity Check
    if args.n < 1:
        raise ValueError("n must be a positive integer.")
    if args.workers < 1:
        raise ValueError("workers must be a positive integer.")
    if args.structure not in params and args.structure != "all":
        raise ValueError(f"Invalid structure: {args.structure}")


    if args.structure == "all":
        structures = params.keys()
    else:
        structures = [args.structure]

    # Simulate the structure(s)
    for structure in structures:
        print(f"Simulating {structure} (x{args.n})...")
        # Retrieve the parameters for the structure
        params_ = params[structure]
        storm_params_ = storm_params[structure]

        # Retrieve the prefix for the files
        if args.o is None:
            prefix = f"./data/{datetime.datetime.now().strftime('%d%m%y-%H%M%S')}/{structure}"
        else:
            prefix = args.o

        # Simulate the structure(s)
        main(
            structure=structure,
            params=params_,
            prefix=prefix,
            ftype=args.ftype,
            n=args.n,
            overwrite=args.overwrite,
            header=args.header,
            workers=args.workers,
            seed=args.seed,
            with_normals=args.with_normals,
            storm_params=storm_params_,
            no_storm=args.no_storm,
        )

        # Save the parameters
        json_dict = {
            'structure_params': params_,
            'storm_params': storm_params_,
            'args': vars(args),
        }
        with open(os.path.join(prefix, f"params_{structure}.json"), "w") as f:
            json.dump(json_dict, f, indent=4)
