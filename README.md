# PointNormalNet
Supplementary code for INF-3990 Master's Thesis: "Reconstructing 3D geometries of sub-cellular structures from SMLM point clouds". Contains the code for the simulator used to generate the training data, as well as the code for the PointNormalNet model. Code for the simulator is in the `simulation` directory, and the inference code for the model is in the `main` directory.

Each directory has its own README file with more details. The README for the simulator is [here](simulation/README.md), and the README for the model is [here](main/README.md).

## Setup
It is required to have an NVIDIA GPU to run the model. The model has been tested on NVIDIA A100-40GB, and takes up 1822 MiB of GPU memory when loaded. The simulator does not require a GPU, but if you have one, it utilises `cupy` to speed up the computation.

Install the dependencies using `pip`, preferably in a virtual environment:
```bash
pip install -r requirements.txt
```
If you have an NVIDIA GPU, you can install `cupy` for faster performance in the simulator:
```bash
pip install cupy
```
I used cupy==11.5.0 for this project. Additionally, you may also need to install the relevant CUDA libraries for your GPU.

## Dataset
The training as well as the test dataset can be downloaded from [here](https://cloud.suyogjadhav.com/index.php/s/LJ6DseSJySwdKYn). Both the datasets are in `.parquet` format and are ready to use for training and testing.