
# Neural Additive Tensor Decomposition for Sparse Tensors
This is a PyTorch implementation of "Neural Additive Tensor Decomposition for Sparse Tensors".
This paper proposed a novel approach that discover non-linear latent patterns in multi-aspect data in an interpretable way.

## Prerequisites
Before you begin using this code, make sure you have the following dependencies installed:
- Python 3.9
- PyTorch
- NumPy
- DotMap


## Usage
To run the demo script, simply execute the run.sh script, which in turn runs src/main.py. 
The script is designed to work with different datasets, which are located in the dataset directory.

## Directory structure

- configs/               # Configuration files for datasets.
- dataset/               # Contains datasets for experimentation.
    - dblp               # Dataset for an inductive setting.
    - trans_dblp         # Dataset for a transductive setting. 
    - ml
    - yelp
    - foursquare_nyc
    - foursquare_tky
    - yahoo_music
- src/
  - main.py              # Running NeAT.
  - read.py              # Reading datasets.
  - model.py             # NeAT model.
  - train.py             # Training NeAT.
  - utils.py             # Saving models.
  - metrics.py           # Evaluation metrics.
- run.sh                 # script for executing the `main.py`
- README.md              # This documentation file.
