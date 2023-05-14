import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from core import Model, PointsGraph, write_ply
import open3d as o3d
from tqdm.auto import tqdm

# Some constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CKPT_DIR = "core/static/weights/strategy1_run2_v2_pgn"
# DEFAULT_CKPT_DIR = "core/static/weights/strategy2_run3_v2_pgn"
MAX_N = 40000
LABEL = "foreground"
NOISE_LABEL = "noise"
NOISE_THRESH = 0.5
DEFAULT_OUTPUT_PATH = "outputs"


def load_model(ckpt_dir: str, device: torch.device = DEVICE):
    """
    Load the model from the checkpoint directory.

    Args:
        ckpt_dir (str): The path to the checkpoint directory.
        device (torch.device): The device to load the model on ('cpu', 'cuda'). Defaults to DEVICE.

    Returns:
        model (Model): The loaded model.
        params (dict): The model parameters.
    """
    # Load the saved parameters
    params = torch.load(os.path.join(ckpt_dir, "model.pth"))

    # Set the seed
    torch.manual_seed(params["misc"]["seed"])

    # Create the model
    model = Model(**params["model"]).to(device)

    # Load the saved weights
    model.load_state_dict(params["state_dict"])
    model.eval()

    del params["state_dict"]  # Save memory

    return model, params


def load_points(path: str, max_n: int = MAX_N, label: str = LABEL, verbose: bool = True):
    """
    Load the points from the given path.

    Args:
        path (str): The path to the points file.
        max_n (int): The maximum number of points to use. Defaults to MAX_N.
        label (str): The label to use. Defaults to LABEL.

    Returns:
        graphs (list): The list of PointsGraphs.
    """
    if path.endswith(".csv"):
        points = pd.read_csv(path)
    elif path.endswith(".tsv"):
        points = pd.read_csv(path, sep="\t")
    elif path.endswith(".parquet"):
        points = pd.read_parquet(path)
    elif path.endswith(".xyz"):
        points = pd.read_csv(path, sep=" ", names=["x", "y", "z"])
    else:
        print("Unsupported file format. Please use csv, tsv or parquet.")
        return None

    graph = to_graph(points, max_n=max_n, label=label, verbose=verbose)
    return graph


def to_graph(
    points: pd.DataFrame, max_n: int = MAX_N, label: str = LABEL, device: str = DEVICE, verbose: bool = True
):
    """
    Convert the points to PointsGraph format.

    Args:
        points (pd.DataFrame): The points.
        max_n (int): The maximum number of points to use. Defaults to MAX_N.
        label (str): The label to use. Defaults to LABEL.
        device (str): The device to use. Defaults to DEVICE.

    Returns:
        graph (PointsGraph): The graph.
    """
    # Figure out the format
    if len(points.columns) == 3:
        pass
    else:
        try:
            points = points[["x", "y", "z"]]
        except:
            try:
                points = points[["X", "Y", "Z"]]
            except:
                try:
                    points = points[["x [nm]", "y [nm]", "z [nm]"]]
                except:
                    try:
                        points = points[["X [nm]", "Y [nm]", "Z [nm]"]]
                    except:
                        print("Could not find xyz columns in the given file.")
                        return None

    # # Bring the points to the right format
    # points.columns = ['x', 'y', 'z']
    # points['label'] = label
    # points['theta'] = 0
    # points['phi'] = 0

    # Check if max_n is exceeded
    points = points.to_numpy()
    n_points = len(points)
    max_n_exceeded = max_n > 0 and len(points) > max_n
    ps = []
    if max_n_exceeded:
        # Split the points into random batches of size max_n
        np.random.shuffle(points)
        pbar = tqdm(total=len(points), leave=False, desc="Loading points")
        while len(points) > max_n:
            pbar.update(1)
            ps.append(points[:max_n])
            points = points[max_n:]
            # ps.append(points.loc[:max_n])
            # # ps.append(points.sample(max_n))
            # points = points.drop(ps[-1].index)
            pbar.update(max_n - 1)

        # Make sure the last batch is not too small
        if len(points) > 0:
            ps.append(np.vstack((ps[-1][: max_n - len(points)], points)))
        pbar.update(len(points))
        pbar.close()

    else:
        ps.append(points)

    # Convert to the PointsGraph format
    graphs = []
    for p in tqdm(ps, leave=False, total=len(ps), desc="Converting to PointsGraph"):
        d = pd.DataFrame(p, columns=["x", "y", "z"])
        d["label"] = label
        d["theta"] = 0
        d["phi"] = 0
        graphs.append(PointsGraph(d, pick_first_n=None, device=device))

    if verbose:
        print(f"Loaded {n_points} points succesfully.")

    return graphs


def infer(
    model: Model,
    graph: PointsGraph,
    strategy: int,
    noise_thresh: float = 0.5,
    device: str = DEVICE,
):
    """
    Perform inference on the given graph.

    Args:
        model (Model): The model.
        graph (PointsGraph): The graph.
        strategy (int): The strategy used (found in the model params).
        noise_thresh (float): The noise threshold. Defaults to 0.5.
        device (str): The device to use. Defaults to DEVICE.

    Returns:
        df (pd.DataFrame): The points with the predicted normals. 6 columns: x, y, z, nx, ny, nz.
    """
    # Perform inference
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            out = model([graph])

    # Retrieve the predicted normals and noise probabilities
    if strategy == 1:
        tpn, norm = out
        _, n = tpn[..., :2], tpn[..., 2:]

    elif strategy == 2:
        normn = out
        norm = normn[..., :3]
        n = normn[..., 3:]

    elif strategy == 3:
        tpn, norm, _ = out
        _, n = tpn[..., :2], tpn[..., 2:]

    # Convert to numpy
    norm = norm.detach().cpu().squeeze().numpy()  # (N, 3)
    n = n.detach().cpu().squeeze().numpy()  # (N, 1)
    is_noise = n > noise_thresh

    # Prepare output
    df = graph.df.copy()
    mod = np.linalg.norm(norm, axis=1)  # Ensure unit normal vectors
    df["nx"] = norm[:, 0] / mod
    df["ny"] = norm[:, 1] / mod
    df["nz"] = norm[:, 2] / mod
    df["label"] = np.where(is_noise, "noise", df["label"])

    return df


def get_3d_mesh(df: pd.DataFrame, depth: int = 8, noise_label: str = NOISE_LABEL, return_densities: bool = False):
    """
    Get the 3D mesh from the given dataframe.

    Args:
        df (pd.DataFrame): The dataframe.
        depth (int): The depth of the mesh. Defaults to 8.
        noise_label (str): The noise label. Defaults to NOISE_LABEL.
        return_densities (bool): Whether to return the densities. Defaults to False.

    Returns:
        mesh (o3d.geometry.TriangleMesh): The mesh.
        pcd (o3d.geometry.PointCloud): The point cloud.
    """
    # Create Open3D point cloud, remove noise
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        df[["x", "y", "z"]][(df["label"] != noise_label)].values
    )
    pcd.normals = o3d.utility.Vector3dVector(
        df[["nx", "ny", "nz"]][(df["label"] != noise_label)].values
    )

    # Create mesh
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    # Add the noisy points back for visualization
    pcd.points = o3d.utility.Vector3dVector(df[["x", "y", "z"]].values)
    pcd.normals = o3d.utility.Vector3dVector(df[["nx", "ny", "nz"]].values)

    # Return the densities if requested
    if not return_densities:
        return mesh, pcd
    else:
        return mesh, pcd, densities


def clean_3d_mesh(mesh, remove_low_quantile, densities=None):
    """
    Clean the 3D mesh by removing low density vertices.

    Args:
        mesh (o3d.geometry.TriangleMesh): The mesh.
        remove_low_quantile (float): The quantile to use for removing low density vertices.
        densities (np.ndarray): The densities. If None, they will be computed. Defaults to None.
    """
    vertices_to_remove = densities < np.quantile(densities, remove_low_quantile)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    # mesh.fill_holes()
    return mesh


def save_3d(
    df: pd.DataFrame, path: str, noise_label: str = NOISE_LABEL, depth: int = 8
):
    """
    Save the inference results to the given path as a 3D reconstruction.
    Can be saved as .ply or .obj. The saved files are in ASCII format for
    easy editing.

    Args:
        df (pd.DataFrame): The dataframe.
        path (str): The path.
        noise_label (str): The label to use for noise points. Defaults to NOISE_LABEL.
        depth (int): The depth to use for the Screened Poisson reconstruction. Defaults to 8.

    Returns:
        mesh (o3d.geometry.TriangleMesh): The mesh.
        pcd (o3d.geometry.PointCloud): The point cloud.
    """

    # Obtain mesh
    mesh, pcd = get_3d_mesh(df, depth=depth, noise_label=noise_label)

    # Save
    o3d.io.write_triangle_mesh(path, mesh, write_vertex_normals=True, write_ascii=True)

    return mesh, pcd


def save_csv(df: pd.DataFrame, path: str):
    """
    Save the inference results to the given path as CSV.

    Args:
        df (pd.DataFrame): The dataframe.
        path (str): The path.
    """
    df.to_csv(path, index=False)


def save_outputs(
    df: pd.DataFrame, path: str, input_path: str, noise_label: str = NOISE_LABEL
):
    """
    Save the inference results to the given path.

    Args:
        df (pd.DataFrame): The dataframe.
        path (str): The path. Must be a directory path.
            If the directory does not exist, it will be created.
        input_path (str): The path to the input file.
        noise_label (str): The label to use for noise points. Defaults to NOISE_LABEL.

    Returns:
        mesh (o3d.geometry.TriangleMesh): The mesh.
        pcd (o3d.geometry.PointCloud): The point cloud.
    """
    # Create the directory if it does not exist
    os.makedirs(path, exist_ok=True)

    # # Save the points and their normals
    # save_csv(df, os.path.join(path, f'{os.path.basename(input_path)}_points.csv'))

    # Save PLY
    mesh, pcd = save_3d(
        df, os.path.join(path, f"{os.path.basename(input_path)}_3D.ply")
    )

    # Save raw PLY
    write_ply(
        df,
        os.path.join(path, f"{os.path.basename(input_path)}_nomesh.ply"),
        noise_label=noise_label,
    )

    return mesh, pcd


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=DEFAULT_CKPT_DIR,
        help="Path to the checkpoint directory. Defaults to core/static/weights/stratefy1_run2_v2_pgn",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the input file (csv/tsv/parquet/xyz with at least three columns: x, y, z). "
        "Header row is optional if only three columns are present. Can automatically detect these common sets of column headers: "
        '["x", "y", "z"], ["X", "Y", "Z"], ["x [nm]", "y [nm]", "z [nm]"], ["X [nm]", "Y [nm]", "Z [nm]"]',
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="A directory path to save the outputs. If not specified, will be saved in the same directory as the input file.",
    )

    parser.add_argument(
        "--noise-thresh",
        type=float,
        default=NOISE_THRESH,
        help="The noise threshold. Defaults to 0.5.",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=MAX_N,
        help="The maximum number of points to use in one batch. Defaults to 100k.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help='The device to use for inference ("cpu", "cuda"). Defaults to "cuda".',
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the resulting 3D reconstruction.",
    )
    args = parser.parse_args()

    if args.max_n < 10000:
        print(
            "WARNING: max-n is set to a low value. This may result in poor performance. Consider increasing it to at least 10000."
        )

    # Load the model
    print(f"Loading model from {args.ckpt_dir}...")
    model, params = load_model(args.ckpt_dir, args.device)
    print("Done!")

    # Load the points
    print(f"Loading points from {args.input}...")
    graphs = load_points(args.input, args.max_n)
    if graphs is None:
        print("Failed to load points. Exiting...")
        exit()
    print("Done!")

    # Perform inference
    print("Performing inference...")
    dfs = []
    for g in tqdm(graphs, leave=False, unit="batch"):
        df = infer(
            model, g, params["model"]["strategy"], args.noise_thresh, args.device
        )
        dfs.append(df)

    df = pd.concat(dfs)

    # Get rid of nans
    df.dropna(inplace=True)

    # Drop duplicates (if any)
    df = df.drop_duplicates(subset=["x", "y", "z"], keep="first")
    print("Done!")

    # Save the outputs
    print(f"Saving outputs to {args.output_path}...")
    mesh, pcd = save_outputs(df, args.output_path, args.input, NOISE_LABEL)
    print("Done!")

    # Visualize
    if args.visualize:
        print("Visualizing...")
        o3d.visualization.draw_geometries([mesh, pcd])
        print("Done!")
