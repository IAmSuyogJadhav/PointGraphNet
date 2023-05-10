import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from core import (
    Model,
    PointsGraph,
    write_ply
)
import open3d as o3d

# Some constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CKPT_DIR = 'core/static/weights/strategy1_run2_v2_pgn'
MAX_N = 100000
LABEL = 'foreground'
NOISE_LABEL = 'noise'
NOISE_THRESH = 0.5
DEFAULT_OUTPUT_PATH = 'outputs'


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
    params = torch.load(os.path.join(ckpt_dir, 'model.pth'))

    # Set the seed
    torch.manual_seed(params['misc']['seed'])

    # Create the model
    model = Model(**params['model']).to(device)

    # Load the saved weights
    model.load_state_dict(params['state_dict'])
    model.half()  #debug
    model.eval()

    del params['state_dict']  # Save memory

    return model, params


def load_points(path: str, max_n: int = MAX_N, label: str = LABEL):
    """
    Load the points from the given path.

    Args:
        path (str): The path to the points file.
        max_n (int): The maximum number of points to use. Defaults to MAX_N.
        label (str): The label to use. Defaults to LABEL.

    Returns:
        points (pd.DataFrame): The points.
    """
    if path.endswith('.csv'):
        points = pd.read_csv(path)
    elif path.endswith('.tsv'):
        points = pd.read_csv(path, sep='\t')
    elif path.endswith('.parquet'):
        points = pd.read_parquet(path)
    else:
        print('Unsupported file format. Please use csv, tsv or parquet.')
        return None
    
    graph = to_graph(points, max_n=max_n, label=label)
    return graph


def to_graph(points: pd.DataFrame, max_n: int = MAX_N, label: str = LABEL, device: str = DEVICE):
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
            points = points[['x', 'y', 'z']]
        except:
            try:
                points = points[['X', 'Y', 'Z']]
            except:
                try:
                    points = points[['x [nm]', 'y [nm]', 'z [nm]']]
                except:
                    try:
                        points = points[['X [nm]', 'Y [nm]', 'Z [nm]']]
                    except:
                        print('Could not find xyz columns in the given file.')
                        return None
                
    # Check if max_n is exceeded
    max_n_exceeded = max_n > 0 and len(points) > max_n
    if max_n_exceeded:
        points = points.sample(n=max_n)
        print(f'Sampling {max_n} points out of {len(points)} total points. Set max_n to -1 to use all points.')

    # Convert to the PointsGraph format
    points.columns = ['x', 'y', 'z']
    points['label'] = label
    points['theta'] = 0
    points['phi'] = 0

    graph = PointsGraph(points, pick_first_n=None, infer_mode=True, device=device)
    print(f'Loaded {len(graph)} points succesfully.')

    return graph


def infer(model: Model, graph: PointsGraph, strategy: int, noise_thresh: float = 0.5, device: str = DEVICE):
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
    df['nx'] = norm[:, 0] / mod
    df['ny'] = norm[:, 1] / mod
    df['nz'] = norm[:, 2] / mod
    df['label'] = np.where(is_noise, 'noise', df['label'])

    return df


def save_ply(df: pd.DataFrame, path: str, noise_label: str = NOISE_LABEL):
    """
    Save the inference results to the given path as PLY.

    Args:
        df (pd.DataFrame): The dataframe.
        path (str): The path.
        noise_label (str): The label to use for noise points. Defaults to NOISE_LABEL.
    """
    # Get rid of noise points
    df = df[df['label'] != noise_label]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(df[['x', 'y', 'z']].values)
    pcd.normals = o3d.utility.Vector3dVector(df[['nx', 'ny', 'nz']].values)

    # Create mesh
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

    # Save
    o3d.io.write_triangle_mesh(path, mesh)


def save_csv(df: pd.DataFrame, path: str):
    """
    Save the inference results to the given path as CSV.

    Args:
        df (pd.DataFrame): The dataframe.
        path (str): The path.
    """
    df.to_csv(path, index=False)    


def save_outputs(df: pd.DataFrame, path: str, input_path: str, noise_label: str = NOISE_LABEL):
    """
    Save the inference results to the given path.

    Args:
        df (pd.DataFrame): The dataframe.
        path (str): The path. Must be a directory path. 
            If the directory does not exist, it will be created.
        input_path (str): The path to the input file.
        noise_label (str): The label to use for noise points. Defaults to NOISE_LABEL.
    """
    # Create the directory if it does not exist
    os.makedirs(path, exist_ok=True)

    # Save the points and their normals
    save_csv(df, os.path.join(path, f'points_{os.path.basename(input_path)}.csv'))

    # Save PLY
    save_ply(df, os.path.join(path, f'{os.path.basename(input_path)}.ply'))

    # Save raw PLY
    write_ply(df, os.path.join(path, f'{os.path.basename(input_path)}_nomesh.ply'), noise_label=noise_label)


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default=DEFAULT_CKPT_DIR, help='Path to the checkpoint directory. Defaults to core/static/weights/stratefy1_run2_v2_pgn')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input file (csv/tsv/parquet with at least three columns: x, y, z). '\
                        'Header row is optional if only three columns are present. Can automatically detect these common sets of column headers: '\
                        '["x", "y", "z"], ["X", "Y", "Z"], ["x [nm]", "y [nm]", "z [nm]"], ["X [nm]", "Y [nm]", "Z [nm]"]')
    parser.add_argument('-o', '--output_path', type=str, default=DEFAULT_OUTPUT_PATH, help='A directory path to save the outputs. If not specified, will be saved in the same directory as the input file.')
    
    parser.add_argument('--noise-thresh', type=float, default=NOISE_THRESH, help='The noise threshold. Defaults to 0.5.')
    parser.add_argument('--max-n', type=int, default=MAX_N, help='The maximum number of points to use for inference. Defaults to 100k.')
    parser.add_argument('--device', type=str, default=DEVICE, help='The device to use for inference ("cpu", "cuda"). Defaults to "cuda".')
    args = parser.parse_args()

    # Load the model
    print(f'Loading model from {args.ckpt_dir}...')
    model, params = load_model(args.ckpt_dir, args.device)
    print('Done!')
    
    # Load the points
    print(f'Loading points from {args.input}...')
    graph = load_points(args.input, args.max_n)
    if graph is None:
        print('Failed to load points. Exiting...')
        exit()
    print('Done!')

    # Perform inference
    print('Performing inference...')
    df = infer(model, graph, params['model']['strategy'], args.noise_thresh, args.device)
    print('Done!')

    # Save the outputs
    print(f'Saving outputs to {args.output_path}...')
    save_outputs(df, args.output_path, args.input, NOISE_LABEL)
    print('Done!')
