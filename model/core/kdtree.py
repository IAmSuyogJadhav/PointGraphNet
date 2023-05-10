import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
# from sklearn.neighbors import KDTree  # Slower than pynanoflann
from pynanoflann import KDTree as KDTree_pnf
from sklearn.neighbors import kneighbors_graph
import os
from scipy.io import loadmat
# from memory_profiler import profile  # uncomment this line, and the @profile lines below to profile memory usage

try:
    import torch
    TORCH_INSTALLED = True
except ImportError:
    print("Torch not installed, training will not work")
    TORCH_INSTALLED = False


class PointsGraph:
    # @profile
    def __init__(self, df, pick_first_n=None, label_format='id', noise_label='noise_bg', no_torch=False, device=None, n_jobs=8, infer_mode=False):
        """Initializes the graph with the given points.

        Args:
            df (pd.DataFrame): The dataframe containing the points.
            pick_first_n (int, optional): If given, only the first n points will be used. Defaults to None.
            label_format (str, optional): The format of the labels. Can be 'id' or 'name'. Defaults to 'id'. Unused.
            noise_label (str, optional): The label of the noise points ['noise_fg', 'noise_bg', 'noise']. Defaults to 'noise_bg'.
                If set to 'noise', all points with a label starting with 'noise' will be considered noise.
            no_torch (bool, optional): If True, a torch copy of the points will not be created. Defaults to False.
            device (torch.device, optional): The device to use for torch. Defaults to None.
            n_jobs (int, optional): Number of jobs to use for pynanoflann kneighbors. Defaults to 8.
            infer_mode (bool, optional): If True, will only expect x, y, and z columns in the dataframe.
                Use for test data. Defaults to False.
        """
        self.df = df.iloc[:pick_first_n]  # Only use the first n points
        self.points = None

        self.points_torch = None  # Points in a torch tensor, used externally for training and such
        self.no_torch = no_torch
        self.device = device
        
        self.noise_label = noise_label 
        self.angle_max = 2 * np.pi # Maximum angle value
        # self.norm = None
        self.gt = None
        self.neighbor_idxs = None  # Cache the indices of the neighbors
        self.neighbor_dists = None  # Cache the distances to the neighbors
        self.last_K = None  # The K used for cached kneighbors calculation
        # self.point_labels = self._get_labels(points=points, format=label_format)
        self.tree = None
        self.n_jobs = n_jobs # Number of jobs to use for pynanoflann kneighbors
        self.infer_mode = infer_mode

        self.reset_tree()  # Initialize the tree
        # Graph (DEPRECATED)
        self.graph = nx.Graph()
        self.edges = []
        self.nodes = []

        # Sanity checks
        # Try to catch if the angle is in degrees and angle max is in radians or vice versa
        if not infer_mode:
            assert (abs(self.df.theta.max() - self.angle_max) <= 100) and self.df.theta.max() <= 2 * self.angle_max,\
                f"Theta max is {self.df.theta.max()}, but should be <= {self.angle_max} (Angle max set in __init__)"
            assert (abs(self.df.phi.max() - self.angle_max) <= 100) and self.df.phi.max() <= 2 * self.angle_max,\
                f"Phi max is {self.df.phi.max()}, but should be <= {self.angle_max} (Angle max set in __init__)"

    def reset_graph(self):
        self.graph = nx.Graph()
        self.nodes = []
        self.edges = []

    def reset_tree(self):
        self.points = self.df[['x', 'y', 'z']].to_numpy()
        if TORCH_INSTALLED and not self.no_torch:
            if not self.infer_mode:
                self.points_torch = torch.from_numpy(self.points).float().to(self.device)
            else:
                self.points_torch = torch.from_numpy(self.points).half().to(self.device)  #debug
        # self.tree = KDTree(self.points)
        self.tree = KDTree_pnf()  # pynaoflann is faster than sklearn
        self.tree.fit(self.points)

    def reset_all(self):
        self.reset_graph()
        self.reset_tree()
    
    # def _get_normals(self, angles: bool=False) -> np.ndarray:
    #     """Returns the normals of the points in the point cloud."""
    #     if self.norm is not None:
    #         if angles and self.norm.shape[1] == 2:
    #             return self.norm
    #         elif not angles and self.norm.shape[1] == 3:
    #             return self.norm
    #     else:
    #         if angles:
    #             self.norm = self.df[['theta', 'phi']].to_numpy()
    #             return self.norm
    #         else:
    #             self.norm = self.df[['nx', 'ny', 'nz']].to_numpy()
    #             return self.norm

    def get_gt(self, strategy=1, K=None) -> tuple[np.ndarray]:
        """Returns the ground truth labels of the poin ts in the point cloud.
        Noise label is set by self.noise_label. The shape is [N, 3] for both the outputs.
        
        Returns:
            tpn_gt: (theta, phi, is_noise) of each point in the point cloud.
            norm_gt: (nx, ny, nz) of each point in the point cloud
            strategy: Which strategy is used for the model.
                1: returns tpn_gt, norm_gt
                2: returns normn_gt
                3: returns tpn_gt, norm_gt, k_nbd_probs_gt
                4: returns normn_gt, instance_ids_gt, instance_points_count_gt
        """
        assert strategy in [1, 2, 3, 4], f"Invalid strategy: {strategy}"
        assert not self.infer_mode, "Cannot get ground truth in infer mode"
        if strategy == 3:
            assert K is not None, "K must be given for strategy 3"

        if self.gt is None:
            if strategy == 1:
                self.df['is_noise'] = (self.df['label'].str.startswith(self.noise_label)).astype(int)
                tpn_gt = self.df[['theta', 'phi', 'is_noise']].to_numpy() / [self.angle_max, self.angle_max, 1]  # Normalize to 0-1
                norm_gt = self.df[['nx', 'ny', 'nz']].to_numpy()

                self.gt = tpn_gt, norm_gt
            elif strategy == 2:
                self.df['is_noise'] = (self.df['label'].str.startswith(self.noise_label)).astype(int)
                normn_gt = self.df[['nx', 'ny', 'nz', 'is_noise']].to_numpy()
                self.gt = normn_gt
            elif strategy == 3: 
                self.df['is_noise'] = (self.df['label'].str.startswith(self.noise_label)).astype(int)
                tpn_gt = self.df[['theta', 'phi', 'is_noise']].to_numpy() / [self.angle_max, self.angle_max, 1]  # Normalize to 0-1
                norm_gt = self.df[['nx', 'ny', 'nz']].to_numpy()
                
                nbd_ids = self.get_k_nearest_all(k=K, include_self=False)[1]
                k_nbd_probs = np.equal(  # [N, K] whether each neighbor belongs to the same instance or not (0/1)
                    self.df.instance_id.values[nbd_ids],
                    self.df.instance_id.values[..., None]
                ).astype(int)

                self.gt = tpn_gt, norm_gt, k_nbd_probs
            elif strategy == 4:
                self.df['is_noise'] = (self.df['label'].str.startswith(self.noise_label)).astype(int)
                normn_gt = self.df[['nx', 'ny', 'nz', 'is_noise']].to_numpy()
                instance_ids_gt = self.df['instance_id'].to_numpy().astype(int)
                
                
                # Mark all the noise points as instance -1
                instance_ids_gt[normn_gt[:, 3] == 1] = -1
                # Add +1 to all the instance ids to make them start from 0
                instance_ids_gt += 1

                # Count the number of points in each instance
                counts = np.bincount(instance_ids_gt)

                # Assign the correct count to each point according to its instance id
                instance_points_count_gt = counts[instance_ids_gt] 

                self.gt = normn_gt, instance_ids_gt, instance_points_count_gt
        
        return self.gt


    def get_k_nearest_all(self, k=5, include_self=True):
        """Returns the k nearest neighbors of all the points in the point cloud."""
        # Check if the neighbors have already been computed for the same k
        if self.neighbor_idxs is None or self.neighbor_dists is None or self.last_K != k:
            assert len(self.points.shape) == 2 and self.points.shape[1] == 3
            # dists, idxs = self.tree.query(self.points, k+1)  # +1 to include the point itself
            dists, idxs = self.tree.kneighbors(self.points, n_neighbors=k+1, n_jobs=self.n_jobs)  # +1 to include the point itself # pynaoflann is faster than sklearn
            idxs = idxs.astype(np.int32)  # Convert uint64 to int32 for compatibility with PyTorch
            self.neighbor_dists, self.neighbor_idxs, self.last_K = dists, idxs, k  # Cache the results
        else:
            dists = self.neighbor_dists  # Retrieve the cached results
            idxs = self.neighbor_idxs

        # Include self or not
        if not include_self:
            # Remove the first column, which contains the distance to the point itself
            return dists[..., 1:], idxs[..., 1:]
        else:
            return dists, idxs
    
    def get_k_nearest(self, idxs, k=5, include_self=True):
        """Returns the k nearest neighbors of the given idxs.
        Does not use the cached results from self.neighbor_*!
        Always computes the neighbors from scratch."""
        points = self.points[idxs].reshape(-1, 3)
        # dists, idxs = self.tree.query(points, k+1)
        dists, idxs = self.tree.kneighbors(points, n_neighbors=k+1, n_jobs=self.n_jobs)  # pynaoflann is faster than sklearn
        idxs = idxs.astype(np.int32)  # Convert uint64 to int32 for compatibility with PyTorch

        if not include_self:
            # Remove the first column, which contains the distance to the point itself
            return dists[..., 1:], idxs[..., 1:]
        else:
            return dists, idxs

    # @profile
    def reduce_points(self, r=2):
        """
        Reduces the number of points by collapsing points that are closer than r.
        A lot of the points will be present in more than one list of neighbors.
        Here's how we deal with that: Sort the list of lists by the length of the list and then start from the top of the list. 
        This way, we prevent removing points that are surrounded by a lot of points 
        (these should be more important than those with less neighbors, as the latter are more ilkely to be edge points).
        At each iteration, the point's value will be replaced with the average position of the list of points in its neighbourhood,
        and all the points in the neigbors' list will be marked as redundant. Any point marked by earlier steps will be ignored in the next steps, 
        and these redundant points will finally be removed at the end.
        """
        # Get all the points within the given radius of each other
        # neighbor_idxs = self.tree.query_radius(self.points, r=r)
        neighbor_idxs = self.tree.radius_neighbors(self.points, radius=r, return_distance=False, n_jobs=self.n_jobs)  # pynaoflann is faster than sklearn

        # Associate the obtained list of neighbors with their corresponding points
        neighbor_idxs = list(map(list, list(neighbor_idxs)))
        point_idxs = list(range(len(neighbor_idxs)))

        # Sort the indices by the length of the list (longest first) and then start from the top of the list.
        # point_idxs.sort(key=lambda i: len(neighbor_idxs[i]), reverse=True)
        point_idxs.sort(key=lambda i: len(neighbor_idxs[i]), reverse=False)  # Sort the indices by the length of the list (shortest first)  # DEBUG
        to_remove = set()
        points = self.points.copy()

        for i in point_idxs:
            # Skip if the point has already been marked for removal
            if i in to_remove:
                continue

            # Get the average position of the list of points in its neighbourhood
            avg_pos = np.mean(points[neighbor_idxs[i]], axis=0)

            # Replace the point's value with the average position
            points[i] = avg_pos

            # Mark the points in the list as redundant except for the point itself
            to_remove.update(set(neighbor_idxs[i]) - {i})

        # Remove the redundant points
        self.points = points[~np.isin(point_idxs, list(to_remove))]

        # Update the tree
        # self.tree = KDTree(self.points)
        self.tree = KDTree_pnf()  # pynaoflann is faster than sklearn
        self.tree.fit(self.points)

        print(f'Removed {len(to_remove)} points.')  # DEBUG

    # @profile
    def _compute_edges(self, k=5):
        node_ends = np.vstack(kneighbors_graph(self.points, n_neighbors=k, include_self=False).tolil().rows)
        node_starts = np.arange(len(self.points))[..., None].repeat(k, axis=1)
        assert len(node_starts) == len(node_ends)  # Sanity check
        self.edges = np.dstack((node_starts, node_ends))
        self.edges = self.edges.reshape(-1, self.edges.shape[-1])

    # @profile
    def _compute_nodes(self):
        self.nodes = np.arange(len(self.points))

    # @profile
    def build_graph(self, k=5):
        # Compute nodes and edges
        self._compute_edges(k=k)
        self._compute_nodes()

        # Add the nodes and edges to the graph
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.edges)

    # @profile
    def visualize(self):
        """Adapted from https://networkx.org/documentation/stable/auto_examples/3d_drawing/plot_basic.html"""

        node_xyz = self.points
        edge_xyz = np.array([(self.points[u], self.points[v]) for u, v in self.graph.edges()])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the nodes
        ax.scatter(*node_xyz.T, s=100, ec='w')

        # Plot the edges
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color="tab:gray")

        # Visual stuff
        ax.grid(False)

        # Suppress tick labels
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_ticks([])

        # Set axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        fig.tight_layout()
        plt.show()

    def show_2d(self, title: str = 'Points (2D)', crop_idxs: tuple = (None, None, None, None), labels: bool=False):
        """Shows a 2D plot of the points.

        Args:
            title (str, optional): The title of the plot. Defaults to 'Points (2D)'.
            crop_idxs (tuple, optional): The indices to crop the plot to, in the format (x_min, x_max, y_min, y_max). Defaults to (None, None, None, None).
            labels (bool, optional): Whether to denote the labels of the points via different colors. Defaults to False.
        """
        points = self.df

        plt.figure(figsize=(10, 5))
        # Plot different labels in different colors
        plt.subplot(1, 2, 1); plt.title('Classes')
        if labels:
            for label in points['label'].unique():
                plt.scatter(points[points['label'] == label]['x'], points[points['label'] == label]['y'], label=label, marker='.', s=2**2)
        else:
            plt.scatter(points['x'], points['y'], marker='.', s=2**2)

        # Crop the plot
        plt.xlim(crop_idxs[0], crop_idxs[1]); plt.ylim(crop_idxs[2], crop_idxs[3])
        plt.legend()

        # Plot different instances in different colors
        plt.subplot(1, 2, 2); plt.title('Instances')

        for instance in points['instance_id'].unique():
            plt.scatter(points[points['instance_id'] == instance]['x'], points[points['instance_id'] == instance]['y'], label=instance, marker='.', s=2**2)

        # Crop the plot
        plt.xlim(crop_idxs[0], crop_idxs[1]); plt.ylim(crop_idxs[2], crop_idxs[3])

        # Set title and show
        plt.suptitle(title)
        plt.show()

    
    # @profile
    def save_graph(self, path):
        nx.set_node_attributes(self.graph, name='x', values=self.df['x'].to_dict())
        nx.set_node_attributes(self.graph, name='y', values=self.df['y'].to_dict())
        nx.set_node_attributes(self.graph, name='z', values=self.df['z'].to_dict())
       
        nx.write_graphml(self.graph, path)

    def build_and_visualize(self, k=5):
        self.build_graph(k=k)
        self.visualize()

    def build_and_save(self, path, k=5):
        self.build_graph(k=k)
        self.save_graph(path)
    
    def __len__(self):
        return len(self.points)
    
    def __getitem__(self, idx):
        return self.points[idx]


# @profile
def load_data(path):
    if path.endswith('.mat') or path.endswith('.dat'):  # For files obtained from testSTORM
        df = pd.read_csv(path, sep='\t', header=None)
        df.columns = ['idx_frame', 'idx_mol', 'x', 'y', 'z', 'N_photon']
        df = df[['x', 'y', 'z']]
    elif path.endswith('.csv'):
        try:
            df = pd.read_csv(path, index_col='id', usecols=['id', 'x', 'y', 'z'])
        except ValueError:  # If the column names are not id, x, y, z, then the file is not in the correct format
            df = pd.read_csv(path, usecols=['x [nm]', 'y [nm]', 'z [nm]'])
            df.rename(columns={'x [nm]': 'x', 'y [nm]': 'y', 'z [nm]': 'z'}, inplace=True)
    return df.astype(np.float16)


def main(k, filepath, last_idx, visualize=False, reduce_radius=None):
    df = load_data(filepath)
    grapher = PointsGraph(df, pick_first_n=last_idx)
    if reduce_radius is not None:
        grapher.reduce_points(reduce_radius)

    if visualize:
        grapher.build_and_visualize(k=k)
    else:
        grapher.build_and_save(f'./graph_{os.path.basename(filepath).split(".")[0]}_k={k}_last_idx={last_idx}_reduce_radius={reduce_radius}.graphml', k=k)


if __name__ == '__main__':
    # seashell_path = './static/particles_2000_processed.csv'
    seashell_path = './static/particles_50000_processed.csv'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--last_idx', type=int, default=-1)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--filepath', type=str, default=seashell_path)
    parser.add_argument('--reduce_radius', type=float, default=None)
    args = parser.parse_args()
    main(args.k, args.filepath, args.last_idx, args.visualize, args.reduce_radius)
