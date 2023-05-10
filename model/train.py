# Imports and Helper Functions
import sys
from collections import OrderedDict
import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core import (
    get_dataset,
    Model,
    get_loss_fn,
    get_metric_fn,
    Dataloader,
    train,
    write_ply,
    infer_on_dataset,
)
from tqdm.auto import tqdm

# Pass "scratch" as an argument to run in dry run mode
if len(sys.argv) > 1 and sys.argv[1] == "scratch":
    scratch_run = True
    print("*" * 80)
    print("NOTE: Running in trial mode (scratch_run = True).")
    print("*" * 80)
else:
    scratch_run = False

# Set the training parameters
STRATEGY = 2
DATA_PATH = ["../simulation/data/v2_pgn"]
PICK_PERCENT = 1.0  # Fraction of data to use (per class, per dataset)
_d = '+'.join([os.path.basename(d) for d in DATA_PATH])
run_prefix = f"strategy{STRATEGY}_metrics_run1_{_d}"
print(run_prefix)

params = {
    "model": {
        "in_channels": 3,  # x, y, z
        "feat_channels": 64,  # N_feat
        "out_channels": 3,  # useless parameter, overriden by strategy
        "K": 4, # number of nearest neighbors to use
        "strategy": STRATEGY,  # 1 = predict theta, phi and calculate norm from them
    },
    "dataset": {
        "data_path": DATA_PATH,
        "classes": ["vesicle", "mito"],
        "max_cache_size": -1,  # -1 = no limit, cache all point clouds
        # "max_cache_size": 0,  # 0 = no cache, each epoch will sample a new set of points
        # "max_cache_size": 10000,  # 10000 = cache 10k point clouds, the cached point clouds will have the same sample of points each epoch, rest will be sampled randomly each epoch
        "sample_size": 10000 if not scratch_run else 1000,  # Number of points to sample from each point cloud
        "replace": True,  # sample with replacement, required when sample_size > size of the smallest point cloud
        "pick_percent": PICK_PERCENT,
        "no_torch": False,  # Faster training with no_torch=False (saves a copy of the point clouds to GPU)
        "device": "cuda",  # "cuda" (for gpu) or "cpu"
        'infer_mode': False,  # Setting to True disable GT functions, set to False for training
        'n_jobs': 16,  # Number of processes to use for pynanoflann kdtree 
    },
    "dataloader": {
        "batch_size": 16,  # Number of point clouds per training batch
        "num_workers": 16,  # Number of workers to use for loading data in parallel
    },
    "training": {
        "val_split": 0.15,  # Fraction of data to use for validation
        "lr": 0.001,  # Learning rate
        "epochs": 100 if not scratch_run else 1,  # Number of epochs to train for
        "momentum": 0.9,  # Momentum for SGD
        "weight_decay": 1e-4,  # Weight decay for SGD
        "scheduler": "cos",  # Learning rate scheduler, "cos" or "none"
        "eta_min": 1e-6,  # for cos
        "save_dir": f"./core/static/weights/{run_prefix}"  # Directory to save checkpoints
        if not scratch_run
        else f"./core/static/weights/{run_prefix}_scratch",
    },
    "loss": {
        "weigh_by_noise_prob": True,  # Weight the loss by the probability of noise
        "normalization": 'exp',  # 'exp', 'log' or 'none'
    },
    "misc": {
        "seed": 42,  # for reproduciblity
    },
}

# Create the save directory
if os.path.exists(params["training"]["save_dir"]) and not scratch_run:
    raise (
        FileExistsError(
            f"Save directory {params['training']['save_dir']} already exists."
        )
    )
else:
    os.makedirs(params["training"]["save_dir"], exist_ok=True)

print(params)

# Set the seed
torch.manual_seed(params["misc"]["seed"])
# np.random.seed(params['misc']['seed'])  # Commented out so that sampling is still random in the dataset

# Create the model
model = Model(**params["model"]).to(params["dataset"]["device"])

# Create the dataset
dataset = get_dataset(**params["dataset"])

if scratch_run:
    dataset.files = dataset.files[:10]

# Split and create the dataloaders
train_size = int((1 - params["training"]["val_split"]) * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, len(dataset) - train_size]
)

train_loader = Dataloader(
    train_dataset,
    **params["dataloader"],
    strategy=params["model"]["strategy"],
    K=params["model"]["K"],
)
val_loader = Dataloader(
    val_dataset,
    **params["dataloader"],
    strategy=params["model"]["strategy"],
    K=params["model"]["K"],
)

# Create the optimizer and scheduler
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=100 * params["training"]["lr"],
    momentum=params["training"]["momentum"],
    weight_decay=params["training"]["weight_decay"],
)
if params["training"]["scheduler"] == "cos":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=params["training"]["epochs"],
        eta_min=params["training"]["eta_min"],
    )
else:
    raise ValueError(f"Unknown scheduler: {params['training']['scheduler']}")

# Create the loss function
loss = get_loss_fn(
    strategy=params["model"]["strategy"], K=params["model"]["K"], **params["loss"]
)

# Create the metric function
def dummy(x, y):  # Use dummy metric (increases training speed)
    return torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
# metric = get_metric_fn(strategy=params["model"]["strategy"], K=params["model"]["K"])
metric = dummy; print('*************** USING DUMMY METRIC ***************')

# Main training loop
train(
    model, optimizer, scheduler, loss, metric, train_loader, val_loader, params
)

# Plot the training history and save
try:
    history = params["history"]
    import matplotlib.pyplot as plt

    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.legend()
    plt.savefig(os.path.join(params["training"]["save_dir"], "loss.png"))
    plt.close()


    plt.plot(history["train_norm_loss"], label="train_norm")
    plt.plot(history["val_norm_loss"], label="val_norm")
    plt.legend()
    plt.savefig(os.path.join(params["training"]["save_dir"], "norm_loss.png"))
    plt.close()

    plt.plot(history["train_tpn_loss"], label="train_tpn")
    plt.plot(history["val_tpn_loss"], label="val_tpn")
    plt.legend()
    plt.savefig(os.path.join(params["training"]["save_dir"], "tp_loss.png"))
    plt.close()

    plt.plot(history["train_noise_loss"], label="train_noise")
    plt.plot(history["val_noise_loss"], label="val_noise")
    plt.legend()
    plt.savefig(os.path.join(params["training"]["save_dir"], "noise_loss.png"))
    plt.close()

    plt.plot(history["train_angle_error"], label="train_angle_error")
    plt.plot(history["val_angle_error"], label="val_angle_error")
    plt.legend()
    plt.savefig(os.path.join(params["training"]["save_dir"], "angle_error.png"))
    plt.close()

    plt.plot(history["train_angle_error_fg"], label="train_angle_error_fg")
    plt.plot(history["val_angle_error_fg"], label="val_angle_error_fg")
    plt.legend()
    plt.savefig(os.path.join(params["training"]["save_dir"], "angle_error_fg.png"))
    plt.close()

    plt.plot(history["train_angle_error_noise"], label="train_angle_error_noise")
    plt.plot(history["val_angle_error_noise"], label="val_angle_error_noise")
    plt.legend()
    plt.savefig(os.path.join(params["training"]["save_dir"], "angle_error_noise.png"))
    plt.close()

except KeyError:
    print("No history to plot")

# Infer on the validation set and save the results
try:
    infer_on_dataset(
        dataset=val_dataset,
        model=model,
        ckpt_dir=params["training"]["save_dir"],
        strategy=params["model"]["strategy"],
        noise_thresh=0.5,
        noise_label="noise_bg",
        n=min(50, len(val_dataset)) if not scratch_run else 1,
    )
except FileNotFoundError:
    print("No checkpoint found, skipping inference...")
