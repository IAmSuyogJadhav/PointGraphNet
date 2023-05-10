import pandas as pd
import os
import glob
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

# Needs to be imported from the root directory of the repo
from . import pointnet2_utils as pnet2
from .kdtree import PointsGraph


# Data loading functions
def load_file(file):
    if file.endswith(".parquet"):
        return pd.read_parquet(file)
    elif file.endswith(".csv"):
        return pd.read_csv(file)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, files, max_cache_size=-1, sample_size=-1, replace=False, no_torch=False, device=None, infer_mode=False, n_jobs=8):
        """Dataset for loading point clouds.

        Args:
            files (list): List of files to load.
            max_cache_size (int, optional): Maximum number of graphs to cache. Defaults to -1 (caching all).
            sample_size (int, optional): Number of points to sample. Defaults to -1 (no sampling).
            replace (bool, optional): Whether to sample with replacement. Defaults to False.
            no_torch (bool, optional): Whether to NOT have the graphs contain torch versions of the point clouds. Defaults to False.
            device (str, optional): Device for torch tensors. Defaults to None.
            infer_mode (bool, optional): Whether to load the point clouds in inference mode (Does not complain about missing gt columns).
                Defaults to False.
            n_jobs (int, optional): Number of jobs for pynanoflann KDTree. Defaults to 8.
        """
        self.files = files

        # Cache
        self.cache = OrderedDict()
        if max_cache_size >= 0:
            self.max_cache_size = max_cache_size
        else:
            self.max_cache_size = len(files)

        # Seed for sampling points
        self.sample_size = sample_size
        self.seed = np.random.randint(0, 2**32 - 1)
        self.replace = (
            replace  # Whether to sample with replacement (allows for oversampling)
        )

        # Misc.
        self.no_torch = no_torch  # Whether to have the graphs contain torch versions of the point clouds
        self.device = device  # Device for torch tensors
        self.infer_mode = infer_mode  # Whether to load the point clouds in inference mode (Does not complain about missing gt columns)
        self.n_jobs = n_jobs  # Number of jobs for pynanoflann KDTree

        # For shuffling
        # self.idxs = np.arange(len(files))

    def reset_seed(self):
        """Resets the seed for sampling points. Run this before each epoch."""
        self.seed = np.random.randint(0, 2**32 - 1)

        # Clear cache
        self.cache.clear()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            # idx = self.idxs[idx]  # To allow shuffling
            # Check if in cache
            if idx in self.cache:
                # graph = self.cache[idx]
                df = self.cache[idx]

            else:  # Load from file (expensive operation)
                file = self.files[idx]
                df = load_file(file)

                # Add to cache, if not full, otherwise don't
                if len(self.cache) < self.max_cache_size:
                    self.cache[idx] = df
                else:
                    pass

            # Sample points
            if self.sample_size > 0:  # No sampling for sample_size <= 0; sampling needed for unequally sized point clouds to form batches
                df = df.sample(
                    self.sample_size, random_state=self.seed, replace=self.replace
                ).reset_index(drop=True)
            
            # Create graph
            graph = PointsGraph(
                df=df,
                pick_first_n=None,
                no_torch=self.no_torch,
                device=self.device,
                infer_mode=self.infer_mode,
                n_jobs=self.n_jobs,
            )
            
            # # Add to cache, if not full, otherwise don't
            # if len(self.cache) < self.max_cache_size:
            #     self.cache[idx] = graph
            # else:
            #     pass

            return graph

        # Support slicing
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        # Support list of indices
        elif isinstance(idx, list):
            return [self[i] for i in idx]


    def __len__(self):
        return len(self.files)

    # def shuffle(self):
    #     """Shuffles the dataset."""
    #     np.random.shuffle(self.idxs)


def get_dataset(data_path, classes, max_cache_size=1000, sample_size=-1, replace=False, pick_percent=1.0, no_torch=False, device=None, infer_mode=False, n_jobs=8):
    """Get pytorch dataset for the given classes and data path.

    Args:
        data_path (str or list): Path to the data folder. Can be a single path or a list of paths.
        classes (list): List of classes to load.
        max_cache_size (int, optional): Maximum number of graphs to cache. Defaults to 1000.
        sample_size (int, optional): Number of points to sample. Defaults to -1 (no sampling).
        replace (bool, optional): Whether to sample with replacement. Defaults to False.
        pick_percent (float, optional): Percent of files to pick, per class, per data_path. Defaults to 1.0.
        no_torch (bool, optional): Whether to NOT have the graphs contain torch versions of the point clouds. Defaults to False.
        device (str, optional): Device for torch tensors. Defaults to None.
        infer_mode (bool, optional): Whether to load the point clouds in inference mode (Does not complain about missing gt columns).
        n_jobs (int, optional): Number of jobs for pynanoflann KDTree. Defaults to 8.
    """
    # Support passing a single path or a list of paths
    if isinstance(data_path, str):
        data_path = [data_path]

    # Get all files
    files = []
    for path in data_path:
        for cls in classes:
            files_ = []
            files_ += glob.glob(os.path.join(f"{path}", f"{cls}_*.parquet"))
            files_ += glob.glob(os.path.join(f"{path}", f"{cls}_*.csv"))
            
            # Pick a percent of files
            files += files_[:int(len(files_) * pick_percent)]
    print(f"Found {len(files)} files.")
    
    return Dataset(
        files=files,
        max_cache_size=max_cache_size,
        sample_size=sample_size,
        replace=replace,
        no_torch=no_torch,
        device=device,
        infer_mode=infer_mode,
        n_jobs=n_jobs,
    )


# Model Definition
class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, strategy=None):
        """PointNet++ model as the encoder.

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 3. Must be >=3.
            out_channels (int, optional): Number of output channels. Defaults to 64.
            strategy (str, optional): Strategy, see the Model description. Defaults to None.

        Modified from: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/
        """
        super().__init__()

        # Original PointNet++ model encoder layers (sem_seg_msg)
        assert in_channels >= 3, "Input channels must be >=3"
        extra_ch = in_channels - 3  # Number of extra channels
        self.sa1 = pnet2.PointNetSetAbstractionMsg(
            1024, [0.05, 0.1], [16, 32], extra_ch, [[16, 16, 32], [32, 32, 64]]
        )
        self.sa2 = pnet2.PointNetSetAbstractionMsg(
            256, [0.1, 0.2], [16, 32], 32 + 64, [[64, 64, 128], [64, 96, 128]]
        )
        self.sa3 = pnet2.PointNetSetAbstractionMsg(
            64, [0.2, 0.4], [16, 32], 128 + 128, [[128, 196, 256], [128, 196, 256]]
        )
        self.sa4 = pnet2.PointNetSetAbstractionMsg(
            16, [0.4, 0.8], [16, 32], 256 + 256, [[256, 256, 512], [256, 384, 512]]
        )
        self.fp4 = pnet2.PointNetFeaturePropagation(512 + 512 + 256 + 256, [256, 256])
        self.fp3 = pnet2.PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = pnet2.PointNetFeaturePropagation(32 + 64 + 256, [256, 128])

        # Modified last layer
        # self.fp1 = pnet2.PointNetFeaturePropagation(128, [128, 128, 128])
        activation = None if strategy == 4 else 'relu'
        self.fp1 = pnet2.PointNetFeaturePropagation(128, [128, 128, 128, out_channels], activation=activation)

    def forward(self, xyz):
        """Forward pass of the model.

        Args:
            xyz (torch.Tensor): Input point cloud, shape (B, 3, N).

        Returns:
            torch.Tensor: Output features, shape (B, `out_channels`. N).
        """
        # Original PointNet++ model encoder pass
        l0_points = None  # From part_seg_msg, when normal_channel is False
        l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        return l0_points


class computeNormals(nn.Module):
    def __init__(self):
        """Compute normals for each point in the point cloud."""
        super().__init__()
        self.pi = torch.tensor(np.pi, dtype=torch.float32, requires_grad=False).cuda()

    def forward(self, tpn):
        """
        Args:
            tpn (torch.Tensor): Input point cloud, shape (B, 3, N). The first
                channel contains the theta value, the second the phi value. The
                third channel contains the probability of it being a noise point.
        """
        # Convert to angles (0-2pi)
        t = tpn[:, 0, :] * 2 * self.pi
        p = tpn[:, 1, :] * 2 * self.pi

        # Compute normals
        out = torch.zeros((tpn.shape[0], 3, tpn.shape[2]), device=tpn.device)
        out[:, 0, :] = torch.cos(t)  # nx
        out[:, 1, :] = torch.sin(t) * torch.sin(p)  # ny
        out[:, 2, :] = torch.sin(t) * torch.cos(p)  # nz

        return out


class Decoder(nn.Module):
    def __init__(self, K=5, in_channels=64, out_channels=4, strategy=1):
        """PointNet++ model as the decoder. The input is the concatenated output
        of the encoder for the point in question and the K neighboring points.
        Therefore, the input has shape (B, (K + 1) * in_channels, N). The output
        has shape (B, out_channels, N).

        Args:
            K (int, optional): Number of nearest neighbors to use. Defaults to 5.
            in_channels (int, optional): Number of channels in the input
                (equal to the `out_channels` of the decoder. Defaults to 64.
            out_channels (int, optional): Number of output channels.
                Defaults to 4 (nx, ny, nz, is_noise). #TODO: Remove this, it's redundant
            strategy (int, optional): Strategy to use for the decoder. 1 is where
                the output is tpn and norm. 2 is where the output is a 4D vector, normn.
                3 is where the output is tpn, norm, and k_nbd_probs. Defaults to 1.
        """
        super().__init__()
        self.K = K
        self.in_channels = in_channels

        if (strategy == 2 or strategy == 4) and out_channels != 4:
            print(
                "Strategies 2 and 4 require out_channels=4 (nx, ny, nz, is_noise). Changing to 4..."
            )
            out_channels = 4
        elif (strategy == 1 or strategy == 3) and out_channels != 3:
            print(
                "Strategy 1 and strategy 3 require out_channels=3 (tpn). Changing to 3..."
            )
            out_channels = 3

        self.out_channels = out_channels
        self.strategy = strategy

        # Model layers (built in a similar way as the original PointNet++ model) (see part_seg_msg)
        self.conv1 = nn.Conv1d((K + 1) * in_channels, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, out_channels, 1)

        if strategy == 3:  # Add a layer to predict the k_nbd_probs
            self.conv3 = nn.Conv1d(128, K, 1)

        # Output Layer
        self.out_activation = nn.Sigmoid()  # Range is [0, 1]
        self.angle2normal = computeNormals()

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input features, shape [B, (K + 1) * in_channels, N].

        Returns:
            torch.Tensor: Normals, will always be in the correct range [-1, 1].
                Shape should be (B, N, 4).
        """
        # Model pass
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)

        # Strat 1: Predict theta, phi, is_noise and then predict normals indirectly
        if self.strategy == 1:
            x = self.conv2(x)  # Shape (B, out_channels, N)
            tpn = self.out_activation(x)  # Theta, phi, and is_noise
            # theta, phi need to be converted to radians by multiplying with 2*pi
            # is_noise is the probability of the point being a noise point
            # Convert to normals (noisy points will have normals of zero)
            norm = self.angle2normal(tpn)

            # Permute to (B, N, 3)
            norm = norm.permute(0, 2, 1)
            tpn = tpn.permute(0, 2, 1)

            return tpn, norm

        # # Strat 2: Predict predict normals directly with an extra channel for is_noise
        elif (self.strategy == 2) or (self.strategy == 4): # Strat 4 is the same as strat 2 but with a different loss
            x = self.conv2(x)  # Shape (B, out_channels, N)
            normn = self.out_activation(x)  # Normals

            # Scale the normals to be in the correct range [-1, 1]
            norm_scaled = normn[:, :3, :] * 2 - 1
            n = normn[:, 3, :]  # Probability of the point being a noise point
            normn = torch.cat((norm_scaled, n.unsqueeze(1)), dim=1)

            # Permute to (B, N, 4)
            normn = normn.permute(0, 2, 1)

            # print(normn.min(), normn.max(), n.min(), n.max(), normn.dtype)
            return normn

        # Strat 3: Same as strat 1 but also predict the probability for each neighbor belonging to the same instance as the point
        elif self.strategy == 3:
            tpn = self.out_activation(self.conv2(x))
            norm = self.angle2normal(tpn)
            k_nbd_probs = self.out_activation(self.conv3(x))

            # Permute to (B, N, 3)
            norm = norm.permute(0, 2, 1)
            tpn = tpn.permute(0, 2, 1)
            k_nbd_probs = k_nbd_probs.permute(0, 2, 1)  # Shape (B, N, K)

            return tpn, norm, k_nbd_probs


class Model(nn.Module):
    def __init__(
        self,
        in_channels=3,
        feat_channels=64,
        out_channels=3,
        K=5,
        strategy=1,
        threshold=0.5,
        device="cuda",
    ):
        """Graph-augmented PointNet++ model. The input is a point cloud graph.
        The output is the normals for each point in the point cloud.

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 3.
            feat_channels (int, optional): Number of channels in the intermediate
                features. Defaults to 64.
            out_channels (int, optional): Number of output channels. Defaults to 3.
            K (int, optional): Number of nearest neighbors to use. Defaults to 5.
            strategy (int, optional): Strategy to use for the decoder. 1 is where
                the output is tpn and norm. 2 is where the output is a 4D vector, normn.
                3 is where the output is tpn, norm, and k_nbd_probs. Defaults to 1.
                4 uses probablistic embeddings.
            threshold (float, optional): Threshold for the probability of a point
                being a noise point. Defaults to 0.5.
            device (str, optional): Device to use. Defaults to 'cuda'.
        """
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels = out_channels
        self.K = K
        self.strategy = strategy
        self.threshold = threshold
        self.device = device

        # Model layers
        self.encoder = Encoder(in_channels=in_channels, out_channels=feat_channels, strategy=strategy).to(
            device
        )

        if strategy == 4:  # Probablistic embeddings
            self.mu_fc = nn.Linear(feat_channels, feat_channels)  # Mean
            self.logvar_fc = nn.Linear(feat_channels, feat_channels)  # Log variance
            self.enc_activation = nn.ReLU()  # Match the activation of the encoder for consistency with other strategies

        self.decoder = Decoder(
            K=K, in_channels=feat_channels, out_channels=out_channels, strategy=strategy
        ).to(device)

    def forward(self, graphs):
        """Forward pass of the model. The noise points are detected by having a high value of
        is_noise (see Decoder). The noise points need to be spotted using a threshold and removed
        from the output.

        Args:
            graphs (List[PointsGraph]): Input point cloud graphs (see PointsGraph).

        Returns: Depending on the `strategy`, one or more of the following outputs are returned.
            torch.Tensor: Theta and Phi values (in the range 0-1, convert to radians before use).
                Also includes the probability of the point being a noise point.
                Shape should be (B, out_channels, N).
            torch.Tensor: Normals, will always be in the correct range [-1, 1].
                Shape should be (B, 3, N).
        """
        # Get the K nearest neighbors for each point (must include the point's own index as well)
        # TODO: Replace this with a version that ignores noise points, possible using strat 4
        nbd_idxs = [
            graph.get_k_nearest_all(k=self.K, include_self=True)[1] for graph in graphs
        ]
        assert all([len(nbd_idx[0]) == self.K + 1 for nbd_idx in nbd_idxs]), (
            "Not all points return K+1 neighbor idxs"
            "Check to make sure that the get_k_nearest_all method is returning the point's own index as well."
        )

        # Input points
        ## Normalize
        inputs = self.pc_normalize_batch(
            torch.stack([graph.points_torch for graph in graphs], dim=0)
        ) # Shape (B, N, in_channels)
        inputs = inputs.to(self.device).permute(0, 2, 1)  # Shape (B, in_channels, N)
        
        # inputs = [
        #     torch.from_numpy(self.pc_normalize(graph.points)).float().to(self.device)
        #     for graph in graphs
        # ]
        # inputs = torch.stack(inputs, dim=0).permute(
        #     0, 2, 1
        # )  # Shape (B, in_channels, N)


        # inputs = torch.cat(
        #     [torch.from_numpy(graph.points[None, ...]).to(self.device) for graph in graphs],
        #     dim=0
        # ).float()  # Shape (B, N, in_channels)

        # ## Normalize the input points and convert to (B, in_channels, N)
        # inputs = self._normalize(inputs).permute(0, 2, 1)  # Shape (B, N, in_channels)
        # print(inputs.shape, inputs.dtype)  # DEBUG

        # Encode
        feats = self.encoder(inputs)  # Shape (B, feat_channels, N)

        if self.strategy == 4:  # Probablistic embeddings
            feats = feats.permute(0, 2, 1)  # Shape (B, N, feat_channels)
            
            # Create a distribution for each point
            mu = F.tanh(self.mu_fc(feats)) + feats  # Shape (B, feat_channels, N)
            logvar = self.logvar_fc(feats)
            # std = torch.exp(logvar / 2)
            # std = torch.clamp(std, min=1e-5)  # Prevents std from being too small
            # assert False, "std is unbounded, so goes all the way to inf. Need to fix it so that it is bounded. sigmoid?"
            std = torch.sigmoid(logvar)  # Bounds the std between 0 and 1
            std = torch.clamp(std, min=1e-5)  # Prevents std from being too small
            q = torch.distributions.Normal(mu, std)

            # Sample from the distribution and apply the activation function
            feats = q.rsample()  # Shape (B, N, feat_channels)
            feats = self.enc_activation(feats)

            # Change back to (B, feat_channels, N) for consistency with other strategies
            feats = feats.permute(0, 2, 1)  # Shape (B, feat_channels, N)

        # Concatenate the features for the K nearest neighbors for each point
        feats_nbd = torch.empty(
            (len(graphs), (self.K + 1) * self.feat_channels, feats.shape[2]),
            device=self.device,
        )
        ## Iterate through the batch of neighborhoods #TODO: This is maybe slow, tried to vectorize but failed
        for i, nbd_idx in enumerate(nbd_idxs):
            # Get the features for the K nearest neighbors of all points
            # print(feats[i, :, nbd_idx].shape)
            feats_nbd[i] = (
                feats[i, :, nbd_idx]
                .permute(0, 2, 1)
                .reshape(  # Shape = (1, K+1, FEAT_CHANNELS)
                    1, (self.K + 1) * self.feat_channels, feats.shape[2]
                )
            )  # Shape = (1, (K+1) * FEAT_CHANNELS, N)

        # # Decode (Strategy 1)
        # tpn, norm = self.decoder(feats_nbd)
        # return tpn, norm

        # # Decode (Strategy 2)
        # norm = self.decoder(feats_nbd)
        # return norm

        # Decode
        out = self.decoder(feats_nbd)

        if self.strategy == 4:
            # Return the distribution parameters as well
            out = (out, mu, std)  # Expects `out` to be NOT a tuple already

        return out

    @staticmethod
    def pc_normalize(pc):
        """
        DEPRECATED:

        Normalize the points as per pointnet++ centroid normalization.
        Borrowed from: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/

        Args:
            pc (np.ndarray): Input points, shape (N, 3).

        Returns:
            np.ndarray: Normalized points, shape (N, 3).
        """
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
    
    @torch.jit.script
    def pc_normalize_batch(pc):
        """Normalize the point cloud as per pointnet++ centroid normalization.
        Converted to a batched torch implementation from original source:
        https://github.com/yanx27/Pointnet_Pointnet2_pytorch/

        Args:
            pc (torch.Tensor): Point cloud, shape (B, N, 3).

        Returns:
            torch.Tensor: Normalized point cloud, shape (B, N, 3).
        """
        centroid = torch.mean(pc, dim=1, keepdim=True)
        pc = pc - centroid
        m = torch.max(
            torch.sqrt(torch.sum(pc ** 2, dim=2, keepdim=True)), dim=1, keepdim=True
        )[0]  # (B, 1, 1); max() returns (values, indices), we only need values so we use [0]
        pc = pc / m
        return pc


def groupby_mean(samples, labels):
    """
    Modified from: https://stackoverflow.com/a/73723767
    """
    assert labels.min() == 0, "Labels must start from 0 for groupby_mean"
    M = torch.zeros(labels.shape[0], labels.max()+1, labels.shape[1]).to(samples.device)
    M[torch.arange(len(labels)).long()[:,None], labels, torch.arange(labels.size(1)).long()] = 1
    M = torch.nn.functional.normalize(M, p=1, dim=-1)
    return M @ samples

def get_kl_criterion(normalization=False, eps=1e-5):
    def _normalised_kl_divergence(p, q):
        """Compute the normalised (optional) KL divergence between two distributions.

        Args:
            p (torch.distributions.Distribution): Distribution 1.
            q (torch.distributions.Distribution): Distribution 2.

        Returns:
            torch.Tensor: KL divergence.
        """
        kl_div = torch.distributions.kl_divergence(p, q)
        if normalization == 'exp':
            # print(f"KL = {kl_div}")
            return (1.0 - torch.exp(-kl_div)).mean()
        elif normalization == 'log':
            # print(f"KL = {kl_div}")
            return (1.0 - torch.log(torch.max(eps, kl_div))).mean()
        elif normalization == 'none':
            # return torch.distributions.kl_divergence(p, q).sum()
            return kl_div.mean()
        else:
            raise ValueError(f"Invalid normalization: {normalization}")
    
    return _normalised_kl_divergence

def _parse_batches(strategy, batch_pred, batch_gt, K=None):
    """Parse the batches for the different strategies."""

    assert strategy in [1, 2, 3, 4], "Invalid strategy"
    
    if strategy == 3:
        assert K is not None, "K must be specified for strategy 3"

    if strategy == 1:
        tpn_pred, norm_pred = batch_pred
        tpn_gt, norm_gt = batch_gt

        # Reshape the tensors to one long vector of Nx3 / Nx1
        tpn_pred = tpn_pred.contiguous().view(-1, 3)
        norm_pred = norm_pred.contiguous().view(-1, 3)
        tpn_gt = tpn_gt.contiguous().view(-1, 3)
        norm_gt = norm_gt.contiguous().view(-1, 3)

        # Separate the noise probability from the theta and phi
        noise_pred = tpn_pred[..., 2].unsqueeze(-1)
        noise_gt = tpn_gt[..., 2].unsqueeze(-1)
        tp_pred = tpn_pred[..., :2]
        tp_gt = tpn_gt[..., :2]

        return tp_pred, tp_gt, norm_pred, norm_gt, noise_pred, noise_gt
    elif strategy == 2:
        normn_pred = batch_pred
        normn_gt = batch_gt

        # Reshape the tensors to one long vector of Nx4
        normn_pred = normn_pred.contiguous().view(-1, 4)
        normn_gt = normn_gt.contiguous().view(-1, 4)

        # Separate the noise probability from the theta and phi
        norm_pred = normn_pred[..., :3]
        norm_gt = normn_gt[..., :3]
        noise_pred = normn_pred[..., 3].unsqueeze(-1)
        noise_gt = normn_gt[..., 3].unsqueeze(-1)

        return norm_pred, norm_gt, noise_pred, noise_gt
    elif strategy == 3:
        tpn_pred, norm_pred, k_nbd_probs = batch_pred
        tpn_gt, norm_gt, k_nbd_probs_gt = batch_gt

        # Reshape the tensors to one long vector of Nx3 / NxK / Nx1
        tpn_pred = tpn_pred.contiguous().view(-1, 3)
        norm_pred = norm_pred.contiguous().view(-1, 3)
        tpn_gt = tpn_gt.contiguous().view(-1, 3)
        norm_gt = norm_gt.contiguous().view(-1, 3)
        k_nbd_probs = k_nbd_probs.contiguous().view(-1, K)
        k_nbd_probs_gt = k_nbd_probs_gt.contiguous().view(-1, K)

        # Separate the noise probability from the theta and phi
        noise_pred = tpn_pred[..., 2].unsqueeze(-1)
        noise_gt = tpn_gt[..., 2].unsqueeze(-1)
        tp_pred = tpn_pred[..., :2]
        tp_gt = tpn_gt[..., :2]

        return tp_pred, tp_gt, norm_pred, norm_gt, noise_pred, noise_gt, k_nbd_probs, k_nbd_probs_gt
    elif strategy == 4:
        normn_pred, mu, std = batch_pred
        normn_gt, instance_ids_gt, instance_points_count_gt = batch_gt

        return normn_pred, normn_gt, mu, std, instance_ids_gt, instance_points_count_gt
    else:
        raise ValueError(f"Unknown strategy {strategy}")
 
def get_loss_fn(K=None, strategy=1, weigh_by_noise_prob=True, normalization='exp'):
    """Get the loss function for the model.
    #TODO: Add weights to the losses

    Args:
        K (int): Number of nearest neighbors used in the model. Needed for strategy 3.
        strategy (int, optional): Strategy used in the model. Defaults to 1.
        weigh_by_noise_prob (bool, optional): Whether to weigh the loss by the noise probability. Defaults to True.
        normalization (str, optional): Normalization to use for the KL divergence. Defaults to 'exp'.
    """
    criterion_tp = nn.CosineEmbeddingLoss()
    criterion_norm = nn.CosineEmbeddingLoss()
    criterion_noise = nn.BCELoss()
    criterion_k_nbd_probs = nn.BCELoss()
    if strategy == 4:
        criterion_kl = get_kl_criterion(normalization=normalization)


    assert strategy in [1, 2, 3, 4], f"Invalid strategy: {strategy}"
    if strategy == 3 and K is None:
        raise ValueError("K must be specified for strategy 3")
   
    def _noise_aware_loss_strat1(batch_pred, batch_gt):
        """Loss function for strategy 1."""
        # print(tpn_pred.shape, tpn_gt.shape)  # DEBUG
        # print(norm_pred.shape, norm_gt.shape)  # DEBUG

        tp_pred, tp_gt, norm_pred, norm_gt, noise_pred, noise_gt = _parse_batches(1, batch_pred, batch_gt)

        # print(noise_pred.shape, noise_gt.shape)  # DEBUG
        # print(tp_pred.shape, tp_gt.shape)  # DEBUG
        # print(tpn_pred.shape, tpn_gt.shape)  # DEBUG
        # print(norm_pred.shape, norm_gt.shape)  # DEBUG

        # Theta and Phi Loss
        # print(tp_pred.shape, noise_pred.shape)  # DEBUG
        if weigh_by_noise_prob:
            tp_pred_noise_weighted = tp_pred * (1 - noise_pred)
            # print(tp_pred.shape, tp_pred_noise_weighted.shape)  # DEBUG
        else:
            tp_pred_noise_weighted = tp_pred

        tp_loss = criterion_tp(
            tp_pred_noise_weighted, tp_gt, torch.ones(tp_gt.shape[0]).cuda()
        )

        # Norm Loss
        if weigh_by_noise_prob:
            norm_pred_noise_weighted = norm_pred * (1 - noise_pred)
            # print(norm_pred.shape, norm_pred_noise_weighted.shape)  # DEBUG
        else:
            norm_pred_noise_weighted = norm_pred
        norm_loss = criterion_norm(
            norm_pred_noise_weighted, norm_gt, torch.ones(norm_gt.shape[0]).cuda()
        )

        # Noise Loss
        noise_probs_loss = criterion_noise(noise_pred, noise_gt)

        # TODO: Add classification at some point maybe

        return tp_loss, norm_loss, noise_probs_loss, torch.tensor(0).cuda(), torch.tensor(0).cuda()

    # def _noise_aware_loss_strat2(normn_pred, normn_gt):
    def _noise_aware_loss_strat2(batch_pred, batch_gt):
        """Loss function for strategy 2."""
        
        norm_pred, norm_gt, noise_pred, noise_gt = _parse_batches(2, batch_pred, batch_gt)

        # print(noise_pred.shape, noise_gt.shape)  # DEBUG
        # print(norm_pred.shape, norm_gt.shape)  # DEBUG

        # Norm Loss
        if weigh_by_noise_prob:
            norm_pred_noise_weighted = norm_pred * (1 - noise_pred)
            # print(norm_pred.shape, norm_pred_noise_weighted.shape)  # DEBUG
        else:
            norm_pred_noise_weighted = norm_pred

        # print(
        #     norm_pred.shape,
        #     norm_gt.shape,
        #     noise_pred.shape,
        #     norm_pred_noise_weighted.shape,
        # )  # DEBUG
        norm_loss = criterion_norm(
            norm_pred_noise_weighted, norm_gt, torch.ones(norm_gt.shape[0]).cuda()
        )

        # Noise Loss
        # print(noise_pred.min(), noise_pred.max(), noise_pred.mean())  # DEBUG
        # noise_pred = torch.clamp(noise_pred, 1e-6, 1 - 1e-6)
        # noise_gt = torch.clamp(noise_gt, 0, 1)
        noise_probs_loss = criterion_noise(noise_pred, noise_gt)

        return (
            torch.tensor(0).cuda(),  # Theta and Phi Loss
            norm_loss,  # Norm Loss
            noise_probs_loss,  # Noise Loss
            torch.tensor(0).cuda(),  # K_NBD Loss
            torch.tensor(0).cuda(),  # KL Loss
        )

    # def _noise_aware_loss_strat3(tpn_pred, norm_pred, k_nbd_probs, tpn_gt, norm_gt, k_nbd_probs_gt):
    def _noise_aware_loss_strat3(batch_pred, batch_gt):
        """Loss function for strategy 3."""
        # print(tpn_pred.shape, tpn_gt.shape)  # DEBUG
        # print(norm_pred.shape, norm_gt.shape)  # DEBUG
        
        tp_pred, tp_gt, norm_pred, norm_gt, noise_pred, noise_gt, k_nbd_probs, k_nbd_probs_gt = _parse_batches(3, batch_pred, batch_gt, K)

        # print(noise_pred.shape, noise_gt.shape)  # DEBUG
        # print(tp_pred.shape, tp_gt.shape)  # DEBUG
        # print(tpn_pred.shape, tpn_gt.shape)  # DEBUG
        # print(norm_pred.shape, norm_gt.shape)  # DEBUG

        # Theta and Phi Loss
        # print(tp_pred.shape, noise_pred.shape)  # DEBUG
        if weigh_by_noise_prob:
            tp_pred_noise_weighted = tp_pred * (1 - noise_pred)
            # print(tp_pred.shape, tp_pred_noise_weighted.shape)  # DEBUG
        else:
            tp_pred_noise_weighted = tp_pred

        tp_loss = criterion_tp(
            tp_pred_noise_weighted, tp_gt, torch.ones(tp_gt.shape[0]).cuda()
        )

        # Norm Loss
        if weigh_by_noise_prob:
            norm_pred_noise_weighted = norm_pred * (1 - noise_pred)
            # print(norm_pred.shape, norm_pred_noise_weighted.shape)  # DEBUG
        else:
            norm_pred_noise_weighted = norm_pred
        norm_loss = criterion_norm(
            norm_pred_noise_weighted, norm_gt, torch.ones(norm_gt.shape[0]).cuda()
        )

        # Noise Loss
        noise_probs_loss = criterion_noise(noise_pred, noise_gt)

        # K-Nearest Neighbors' probabilities (of belonging to the same instance as the point) loss
        k_nbd_probs_loss = criterion_k_nbd_probs(k_nbd_probs, k_nbd_probs_gt)

        return tp_loss, norm_loss, noise_probs_loss, k_nbd_probs_loss, torch.tensor(0).cuda()

    def _noise_aware_loss_strat4(batch_pred, batch_gt):
        """Loss function for strategy 4."""

        normn_pred, normn_gt, mu, std, instance_ids_gt, instance_points_count_gt = _parse_batches(4, batch_pred, batch_gt)

        # Get the strategy 2 loss
        tp_loss, norm_loss, noise_probs_loss, k_nbd_probs_loss, _ = _noise_aware_loss_strat2(
            normn_pred, normn_gt
        )
        
        # with torch.no_grad():  # No need to compute gradients for this
        # Get average distribution parameters for each instance
        instance_average_mu = groupby_mean(mu, instance_ids_gt)  # batch_size, n_instances, n_feats
        
        # std is a bit more complicated, but basically, std_average = sqrt(mean(var / instance_points_count)) = sqrt(sum(std^2)) / instance_points_count
        eps = 1e-8  # To avoid sqrt zero, which has infinite gradient
        instance_average_std = groupby_mean((std ** 2) / instance_points_count_gt.unsqueeze(-1), instance_ids_gt)  # batch_size, n_instances, n_feats
        instance_average_std = torch.sqrt(torch.abs(instance_average_std) + eps)  # batch_size, n_instances, n_feats

        # Assign instance (mu, std) to points based on their instance_ids
        instance_average_mu = instance_average_mu[
            torch.arange(instance_ids_gt.shape[0]).long().unsqueeze(-1), instance_ids_gt]  # batch_size, n_points, n_feats
        instance_average_std = instance_average_std[
            torch.arange(instance_ids_gt.shape[0]).long().unsqueeze(-1), instance_ids_gt]  # batch_size, n_points, n_feats

        # Instances' normal distribution p
        p = torch.distributions.Normal(instance_average_mu, instance_average_std)
        
        # Points' normal distribution q
        q = torch.distributions.Normal(mu, std)

        # KL Divergence loss between p and q, should be close to 0 for points belonging to the same instance
        kl_loss_pos = criterion_kl(q, p)  # KL divergence between positive examples

        kl_loss_neg = torch.tensor(0.0).cuda()  # KL divergence between negative examples  
        # print('[WARN] Not calculating KL divergence loss for negative examples for now')
        #DEBUG : Not calculating KL divergence loss for negative examples for now 
        
        _count = instance_average_mu.shape[0]
        for instance_mu, instance_std, instance_id in zip(instance_average_mu, instance_average_std, instance_ids_gt):
            # Broadcast instance indices for faster computation
            total_instances = instance_id.max().item() + 1  # Total number of instances, including the noise instance (0)
            all = list(range(0, total_instances))  # List of all the instances, including the noise instance (0)

            instances = [[all[i]] * (total_instances - 1) for i in range(1, total_instances)]  # All instance IDs, excluding the noise instance (0)
            handshakes = [all[i+1:] + all[:i] for i in range(1, len(all))]  # All possible comparisons with the rest of the instance IDs, including the noise instance
            
            # Distributions of all the instances, excluding the noise instance
            p = torch.distributions.Normal(instance_mu[instances,], instance_std[instances,])
            
            # Compare with the rest of the instances, including the noise instance
            q = torch.distributions.Normal(instance_mu[handshakes,], instance_std[handshakes,])

            # Compute the KL divergence between the distributions
            kl_loss_neg += criterion_kl(q, p)

        
        # ######################### NAIVE IMPLEMENTATION START #########################
        # _count = 0
        # normal_distributions = [
        #     [torch.distributions.Normal(instance_average_mu[b, i], instance_average_std[b, i]) for i in range(torch.unique(instance_ids_gt[b]).shape[0])]
        #     for b in range(mu.shape[0])
        # ]
        # for b in range(mu.shape[0]):  # For each sample in the batch
        #     for i in range(torch.unique(instance_ids_gt[b]).shape[0]):  # For each instance in the sample
        #         # Current instance's distribution p
        #         # with torch.no_grad():
        #         # p = torch.distributions.Normal(instance_average_mu[b, i], instance_average_std[b, i])
        #         p = normal_distributions[b][i]
        #         # Compare with other instances' distributions q
        #         for j in range(torch.unique(instance_ids_gt[b]).shape[0]):
        #             if i != j:
        #                 q = torch.distributions.Normal(instance_average_mu[b, j], instance_average_std[b, j])
        #                 kl_loss_neg += criterion_kl(q, p)
        #                 _count += 1
        
        # ######################### NAIVE IMPLEMENTATION END #########################

        # Average the KL divergence loss
        kl_loss_neg /= _count

        # KL Divergence loss total
        kl_loss = kl_loss_pos - kl_loss_neg + 1  # +1 to make sure it's always >= 0 like other losses
        # minimising kl_loss_pos and maximising kl_loss_neg should be equivalent to minimising kl_loss
        
        return tp_loss, norm_loss, noise_probs_loss, k_nbd_probs_loss, kl_loss

    if strategy == 1:
        return _noise_aware_loss_strat1
    elif strategy == 2:
        return _noise_aware_loss_strat2
    elif strategy == 3:
        return _noise_aware_loss_strat3
    elif strategy == 4:
        return _noise_aware_loss_strat4

def get_metric_fn(strategy=1, K=None):
    
    assert strategy in [1, 2, 3, 4], 'Invalid strategy'
    if strategy == 3:
        assert K is not None, 'K must be specified for strategy 3'
    
    # criterion_norm = nn.CosineEmbeddingLoss(reduction='none')
    criterion_norm = nn.CosineSimilarity(dim=-1)

    def _angle_error(batch_pred, batch_gt):
        """Compute the angle error in degrees.

        Args:
            batch_pred (torch.Tensor): Predicted batch outputs.
            batch_gt (torch.Tensor): Ground truth batch outputs.

        Returns:
            angle, angle_fg, angle_noise: Angle error, shape (batch_size,).
                Angle error for foreground points, shape (batch_size,).
                Angle error for noise points, shape (batch_size,).
        """
        parsed = _parse_batches(strategy, batch_pred, batch_gt, K)
        if strategy == 1:
            _, _, norm_pred, norm_gt, _, noise_gt = parsed
        elif strategy == 2:
            norm_pred, norm_gt, _, noise_gt = parsed
        elif strategy == 3:
            _, _, norm_pred, norm_gt, _, noise_gt, _, _ = parsed
        elif strategy == 4:
            normn_pred, normn_gt, _, _, _, _ = parsed
            norm_pred, norm_gt, _, noise_gt = _parse_batches(2, normn_pred, normn_gt)  # strat2
        else:
            raise NotImplementedError

        # Norm Loss
        norm_loss = criterion_norm(
            norm_pred, norm_gt,
            # torch.ones(norm_gt.shape[0]).cuda()
        )

        # Norm loss for foreground points (should be close to 1)
        norm_pred_fg_weighted = norm_pred * (1 - noise_gt)
        norm_loss_fg_weighted = criterion_norm(
            norm_pred_fg_weighted, norm_gt,
            # torch.ones(norm_gt.shape[0]).cuda()
        )

        # Norm loss for noise points (should be close to 0)
        norm_pred_noise = norm_pred * noise_gt
        norm_loss_noise = criterion_norm(
            norm_pred_noise, norm_gt * noise_gt,
            # torch.ones(norm_gt.shape[0]).cuda()
        )

        # Angle error (in radians)
        # angle = torch.acos(1 - norm_loss)
        # angle_fg = torch.acos(1 - norm_loss_fg_weighted)
        # angle_noise = torch.acos(1 - norm_loss_noise)
        angle = torch.acos(norm_loss)
        angle_fg = torch.acos(norm_loss_fg_weighted)
        angle_noise = torch.acos(norm_loss_noise)

        # Angle error (in degrees)
        angle = angle.mean() * 180 / np.pi
        angle_fg = angle_fg.mean() * 180 / np.pi
        angle_noise = angle_noise.mean() * 180 / np.pi

        return angle, angle_fg, angle_noise

    return _angle_error


# Training and Validation stuff
class Dataloader(torch.utils.data.DataLoader):
    def __init__(self, dataset, strategy=1, K=None, **kwargs):
        """Custom dataloader to return the ground truth as well.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to use.
            strategy (int): Strategy to use. 1, 2, 3 or 4.
            K (int): Number of nearest neighbors to use for strategy 3.

        Yields:
            tuple: (batch, ground_truth), where ground_truth is
                a tuple of (tpn, norm) for strategy 1,
                normn for strategy 2, and
                (tpn, norm, k_nbd_probs) for strategy 3.
                (normn, instance_ids, instance_points_count) for strategy 4.
        """
        super().__init__(dataset, **kwargs)
        self.strategy = strategy
        self.K = K

        assert strategy in [1, 2, 3, 4], "Strategy must be 1, 2, 3 or 4."
        if strategy == 3:
            assert K is not None, "K must be specified for strategy 3."

    def __iter__(self):

        for i in range(0, len(self.dataset), self.batch_size):
            batch = self.dataset[i : i + self.batch_size]
            batch_gt = [d.get_gt(strategy=self.strategy, K=self.K) for d in batch]
            if self.strategy == 1:
                batch_tpns, batch_norms = zip(*batch_gt)
                batch_tpns = torch.from_numpy(np.stack(batch_tpns)).float().cuda()
                batch_norms = torch.from_numpy(np.stack(batch_norms)).float().cuda()
                gt = (batch_tpns, batch_norms)
            elif self.strategy == 2:
                batch_normns = torch.from_numpy(np.stack(batch_gt)).float().cuda()
                gt = batch_normns
            elif self.strategy == 3:
                batch_tpns, batch_norms, batch_k_nbd_probs = zip(*batch_gt)
                batch_tpns = torch.from_numpy(np.stack(batch_tpns)).float().cuda()
                batch_norms = torch.from_numpy(np.stack(batch_norms)).float().cuda()
                batch_k_nbd_probs = (
                    torch.from_numpy(np.stack(batch_k_nbd_probs)).float().cuda()
                )
                gt = (batch_tpns, batch_norms, batch_k_nbd_probs)
            elif self.strategy == 4:
                batch_normns, batch_instance_ids, batch_instance_points_count = zip(*batch_gt)
                batch_normns = torch.from_numpy(np.stack(batch_normns)).float().cuda()
                batch_instance_ids = torch.from_numpy(np.stack(batch_instance_ids)).long().cuda()
                batch_instance_points_count = torch.from_numpy(np.stack(batch_instance_points_count)).long().cuda()
                gt = (batch_normns, batch_instance_ids, batch_instance_points_count)
            # yield batch, batch_tpns, batch_norms
            yield batch, gt


def train(model, optimizer, scheduler, loss_fn, metric_fn, train_loader, val_loader, params):
    # DEBUG
    torch.autograd.set_detect_anomaly(True)

    try:
        # Training history
        history = {
            # Losses
            "train_norm_loss": [],
            "train_tpn_loss": [],
            "train_noise_loss": [],
            "train_k_nbd_probs_loss": [],
            "train_kl_loss": [],
            "train_loss": [],
            "val_norm_loss": [],
            "val_tpn_loss": [],
            "val_noise_loss": [],
            "val_k_nbd_probs_loss": [],
            "val_kl_loss": [],
            "val_loss": [],

            # Metrics
            "train_angle_error": [],
            "train_angle_error_fg": [],
            "train_angle_error_noise": [],
            "val_angle_error": [],
            "val_angle_error_fg": [],
            "val_angle_error_noise": [],

            # Best epoch
            "best_val_loss": np.inf,
            "best_epoch": 0,
        }

        # Training Loop
        best_val_loss = np.inf
        for epoch in tqdm(
            range(params["training"]["epochs"]),
            total=params["training"]["epochs"],
            desc="Training",
        ):
            # Train
            model.train()

            count = 0
            train_norm_loss = 0
            train_tpn_loss = 0
            train_noise_loss = 0
            train_k_nbd_probs_loss = 0
            train_kl_loss = 0
            train_loss = 0
            train_angle_error = 0
            train_angle_error_fg = 0
            train_angle_error_noise = 0

            # for i, (batch, batch_tpns, batch_norms) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Train', leave=False):
            for i, (batch, batch_gt) in tqdm(
                enumerate(train_loader), total=len(train_loader), desc="Train", leave=False
            ):
                optimizer.zero_grad()

                # Get the model output
                # batch_tpns_pred, batch_norms_pred = model(batch)
                batch_pred = model(batch)

                # Calculate the loss
                # tpn_loss, norm_loss, noise_loss, k_nbd_probs_loss = loss_fn(batch_tpns_pred, batch_norms_pred, batch_tpns, batch_norms)
                tpn_loss, norm_loss, noise_loss, k_nbd_probs_loss, kl_loss = loss_fn(
                    batch_pred, batch_gt
                )
                loss = norm_loss + tpn_loss + noise_loss + k_nbd_probs_loss + kl_loss

                # Backprop
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1?) # Gradient clipping if needed
                optimizer.step()

                # Update the losses
                train_norm_loss += norm_loss.item() * len(batch)
                train_tpn_loss += tpn_loss.item() * len(batch)
                train_noise_loss += noise_loss.item() * len(batch)
                train_k_nbd_probs_loss += k_nbd_probs_loss.item() * len(batch)
                train_kl_loss += kl_loss.item() * len(batch)
                train_loss += loss.item() * len(batch)
                count += len(batch)

                # Calculate and update the metrics
                # with torch.no_grad():
                angle_error, angle_error_fg, angle_error_noise = metric_fn(batch_pred, batch_gt)
                train_angle_error += angle_error.item() * len(batch)
                train_angle_error_fg += angle_error_fg.item() * len(batch)
                train_angle_error_noise += angle_error_noise.item() * len(batch)

                # tqdm.write(f"Batch {i+1}/{len(train_loader)}: train_loss={train_loss/(i+1):.4f}")

            # Update the history
            history["train_norm_loss"].append(train_norm_loss / count)
            history["train_tpn_loss"].append(train_tpn_loss / count)
            history["train_noise_loss"].append(train_noise_loss / count)
            history["train_k_nbd_probs_loss"].append(train_k_nbd_probs_loss / count)
            history["train_kl_loss"].append(train_kl_loss / count)
            history["train_loss"].append(train_loss / count)

            history["train_angle_error"].append(train_angle_error / count)
            history["train_angle_error_fg"].append(train_angle_error_fg / count)
            history["train_angle_error_noise"].append(train_angle_error_noise / count)

            # Print the epoch train loss
            tqdm.write(
                f"Epoch {epoch+1}/{params['training']['epochs']}: train_loss={train_loss/count:.4f} "
                f"[norm={train_norm_loss/count:.4f}, tpn={train_tpn_loss/count:.4f}, noise={train_noise_loss/count:.4f}, "
                f"k_nbd_probs={train_k_nbd_probs_loss/count:.4f}, kl={train_kl_loss/count:.4f}] "\
                f"[angle_error={train_angle_error/count:.4f}, angle_error_fg={train_angle_error_fg/count:.4f}, angle_error_noise={train_angle_error_noise/count:.4f}]"        
            )

            # Validate
            model.eval()

            count = 0
            val_norm_loss = 0
            val_tpn_loss = 0
            val_noise_loss = 0
            val_k_nbd_probs_loss = 0
            val_kl_loss = 0
            val_loss = 0
            val_angle_error = 0
            val_angle_error_fg = 0
            val_angle_error_noise = 0
            # for i, (batch, batch_tpns, batch_norms) in tqdm(enumerate(val_loader), total=len(val_loader), desc='Val', leave=False):
            for i, (batch, batch_gt) in tqdm(
                enumerate(val_loader), total=len(val_loader), desc="Val", leave=False
            ):
                # Get the model output
                # batch_tpns_pred, batch_norms_pred = model(batch)
                batch_pred = model(batch)

                # Calculate the loss
                # tpn_loss, norm_loss, noise_loss, k_nbd_probs_loss = loss_fn(batch_tpns_pred, batch_norms_pred, batch_tpns, batch_norms)
                tpn_loss, norm_loss, noise_loss, k_nbd_probs_loss, kl_loss = loss_fn(
                    batch_pred, batch_gt
                )
                loss = norm_loss + tpn_loss + noise_loss + k_nbd_probs_loss + kl_loss

                # Update the losses
                val_norm_loss += norm_loss.item() * len(batch)
                val_tpn_loss += tpn_loss.item() * len(batch)
                val_noise_loss += noise_loss.item() * len(batch)
                val_k_nbd_probs_loss += k_nbd_probs_loss.item() * len(batch)
                val_kl_loss += kl_loss.item() * len(batch)
                val_loss += loss.item() * len(batch)
                count += len(batch)

                # Calculate and update the metrics
                # with torch.no_grad():
                angle_error, angle_error_fg, angle_error_noise = metric_fn(batch_pred, batch_gt)
                val_angle_error += angle_error.item() * len(batch)
                val_angle_error_fg += angle_error_fg.item() * len(batch)
                val_angle_error_noise += angle_error_noise.item() * len(batch)

                # tqdm.write(f"Batch {i+1}/{len(val_loader)}: val_loss={val_loss/(i+1):.4f}")

            # Update the history
            history["val_norm_loss"].append(val_norm_loss / count)
            history["val_tpn_loss"].append(val_tpn_loss / count)
            history["val_noise_loss"].append(val_noise_loss / count)
            history["val_k_nbd_probs_loss"].append(val_k_nbd_probs_loss / count)
            history["val_kl_loss"].append(val_kl_loss / count)
            history["val_loss"].append(val_loss / count)

            history["val_angle_error"].append(val_angle_error / count)
            history["val_angle_error_fg"].append(val_angle_error_fg / count)
            history["val_angle_error_noise"].append(val_angle_error_noise / count)

            # Print the epoch val loss
            tqdm.write(
                f"Epoch {epoch+1}/{params['training']['epochs']}: val_loss={val_loss/count:.4f} "
                f"[norm={val_norm_loss/count:.4f}, tpn={val_tpn_loss/count:.4f}, noise={val_noise_loss/count:.4f}, "
                f"k_nbd_probs={val_k_nbd_probs_loss/count:.4f}, kl={val_kl_loss/count:.4f}] "\
                f"[angle_error={val_angle_error/count:.4f}, angle_error_fg={val_angle_error_fg/count:.4f}, angle_error_noise={val_angle_error_noise/count:.4f}]"
            )

            # Update the scheduler
            scheduler.step()

            # Save the model if it's the best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                history["best_val_loss"] = best_val_loss
                history["best_epoch"] = epoch
                params["history"] = history
                params["state_dict"] = model.state_dict()

                torch.save(
                    params, os.path.join(params["training"]["save_dir"], "model.pth")
                )
                tqdm.write(
                    f"Epoch {epoch+1}/{params['training']['epochs']}: Saved model with val_loss={val_loss/count:.4f}"
                )
    except KeyboardInterrupt:
        print("Training interrupted! Stopping...")
    print("Training finished!")
    print(f"Best val_loss={best_val_loss/(count + 1e-8):.4f} at epoch {history['best_epoch']+1}")


# Inference tools
def write_ply(df, fname, noise_label="noise_bg"):
    """Write a dataframe to a ply file."""
    # Background noise normals set to 0
    df.loc[df.label.str.startswith(noise_label), "ny"] = 0
    df.loc[df.label.str.startswith(noise_label), "nx"] = 0
    df.loc[df.label.str.startswith(noise_label), "nz"] = 0

    # Convert parquet to XYZ
    parq_df = df[["x", "y", "z", "nx", "ny", "nz"]].copy()

    # Create ply header
    ply_header = """ply
    format ascii 1.0
    element vertex {}
    property float x
    property float y
    property float z
    property float nx
    property float ny
    property float nz
    end_header
    """

    # Write ply
    with open(fname, "w") as f:
        f.write(ply_header.format(len(parq_df)))
        np.savetxt(f, parq_df, "%f %f %f %f %f %f")


def infer_on_dataset(
    dataset,
    model,
    ckpt_dir,
    strategy=1,
    noise_thresh=0.5,
    noise_label="noise_bg",
    n=None,
    save_df=False,
):
    # Infer on all points without sampling
    if isinstance(dataset, torch.utils.data.Subset):
        try:
            dataset.dataset.cache.clear()
            dataset.dataset.sample_size = -1
        except AttributeError:  # If the dataset is a subset of a subset
            dataset.dataset.dataset.cache.clear()
            dataset.dataset.dataset.sample_size = -1
    else:
        dataset.cache.clear()
        dataset.sample_size = -1

    # Sample n point clouds from the dataset
    if n is not None:
        dataset = torch.utils.data.Subset(
            dataset, np.random.choice(len(dataset), n, replace=False)
        )

    # Load best model weights
    model.load_state_dict(
        torch.load(os.path.join(f"{ckpt_dir}", "model.pth"))["state_dict"]
    )
    model.eval()

    # Infer on the validation set
    pred_norms = []
    pred_tps = []
    pred_is_noises = []

    for i in tqdm(range(len(dataset)), desc="Infering on dataset"):
        if strategy == 1:
            tpn, norm = model([dataset[i]])
            tp, n = tpn[..., :2], tpn[..., 2:]

        elif strategy == 2:
            normn = model([dataset[i]])
            norm = normn[..., :3]
            n = normn[..., 3:]
            tp = torch.zeros((n.shape[0], 2))  # Dummy tensor

        elif strategy == 3:
            tpn, norm, k_nbd_probs = model([dataset[i]])
            tp, n = tpn[..., :2], tpn[..., 2:]
            print(
                f"NOTE TO SELF: TODO: Gather instance IDs from k_nbd_probs "
                "using a greedy algorithm and save them to the ply file as different colors."
            )
        elif strategy == 4:
            normn, _, _ = model([dataset[i]])  # returns normn, mu, std
            norm = normn[..., :3]
            n = normn[..., 3:]
            tp = torch.zeros((n.shape[0], 2))  # Dummy tensor

        pred_norms.append(norm.detach().cpu().squeeze().numpy())
        pred_tps.append(tp.detach().cpu().squeeze().numpy())
        pred_is_noises.append((n > noise_thresh).detach().cpu().squeeze().numpy())

    # Save the predictions and the ground truth as ply files
    save_dir = os.path.join(ckpt_dir, "inference")
    os.makedirs(save_dir, exist_ok=True)
    for i in tqdm(range(len(dataset)), desc="Saving ply files"):
        # Save the ground truth
        df = dataset[i].df
        label = df.label[~df.label.str.startswith('noise')][0]
        write_ply(df, os.path.join(save_dir, f"{label}_{i}_gt.ply"), noise_label=noise_label)
        
        # Save the df
        if save_df:
            df.to_parquet(os.path.join(save_dir, f"{label}_{i}_gt.parquet"))

        # Save the predictions
        df["nx"] = pred_norms[i][:, 0]
        df["ny"] = pred_norms[i][:, 1]
        df["nz"] = pred_norms[i][:, 2]
        # print(df.label.shape, pred_is_noises[i].shape, np.where(pred_is_noises[i], noise_label, 'foreground').shape)
        df["label"] = np.where(pred_is_noises[i], noise_label, "foreground")
        write_ply(df, os.path.join(save_dir, f"{label}_{i}_pred.ply"), noise_label=noise_label)

        # Save the df
        if save_df:
            df.to_parquet(os.path.join(save_dir, f"{label}_{i}_pred.parquet"))
