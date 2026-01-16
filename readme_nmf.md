# NMF Project – Advanced Machine Learning

## Project Description

This project implements several variants of Non-negative Matrix Factorization (NMF) for image analysis and reconstruction (MNIST, Flowers102).
A central objective is to study the low-rank bias phenomenon — the tendency of models to first learn simple / low-frequency structures before capturing finer details.

## Scientific Objectives

- Implement NMF using:
  - Gradient descent
  - Multiplicative updates
- Implement Deep NMF (multi-layer factorization)
- Study low-rank bias through effective rank analysis
- Image reconstruction and generation
- Early stopping based on effective-rank plateau detection

## Project Structure

nmf_project/
|-- data_loader.py        # Dataset loading (MNIST, Flowers102)
|-- metrics.py            # Metrics (effective rank, nuclear norm, etc.)
|-- nmf_models.py         # NMF and Deep NMF models
|-- visualizations.py     # Visualization and plotting utilities
|-- experiments.py        # Main experiment orchestration script
`-- readme_nmf.md         # This file


## File Descriptions

### data_loader.py

Handles dataset loading and preprocessing for NMF.

- load_mnist(resize=(28, 28))
  Loads MNIST and returns the flattened data matrix X.

- load_flowers102(resize=(64, 64))
  Loads the Flowers102 dataset.

### metrics.py

Metric functions used to analyze matrix factorizations.

- exp_effective_rank_torch(A)
  Computes the exponential effective rank (diversity of singular values).

- nuclear_over_operator_norm_torch(A)
  Ratio of nuclear norm to operator norm.

- cosine_separation_loss(H)
  Penalty encouraging orthogonality between rows of H.

### nmf_models.py

Implementations of various NMF variants.

- Deep_NMF_2W(A, r1, r2, init, end, epochs)
  Two-layer Deep NMF:
  A ≈ W1 @ W2 @ H

- NMF_for_r_comparison(A, r, init, end, epochs)
  Classical NMF using gradient descent (Adam).

- NMF_for_r_comparison_MU(A, r, init, end, epochs)
  NMF using multiplicative updates (Lee & Seung).

### visualizations.py

Visualization utilities.

- plot_nmf_results(W, H, ...)
  Heatmaps of factor matrices and training curves (error, rank, singular values).

- plot_H_signatures(H, title, ...)
  Displays rows of H as images (NMF signatures).

- plot_mnist_reconstruction(A, W1, W2, H, ...)
  Original vs reconstructed image (Deep NMF).

- plot_mnist_reconstruction_nmf(A, W, H, ...)
  Original vs reconstructed image (standard NMF).

### experiments.py

Main script that orchestrates training, evaluation, and visualization.

## Installation and Usage

### Requirements

pip install numpy pandas matplotlib seaborn tqdm scikit-learn scipy torch torchvision

### Quick Start

1. Copy all files into the same directory

2. Run:
   python experiments.py

3. Results are saved automatically in a timestamped folder, for example:
   C:\Users\thoma\Desktop\Code\NMF Graphiques\Advanced ML\20250106_143025\

## Configuration

### Switching Dataset

In experiments.py, replace:

X, dataset = load_mnist(resize=(28, 28))

with:

X, dataset = load_flowers102(resize=(64, 64))

### Hyperparameters Example

W1, W2, H, errorsGD, rankGD, nuclearrankGD, SVGD1, SVGD2 = Deep_NMF_2W(
    X[100:350, :],
    r1=20,
    r2=10,
    init='random',
    end='all',
    epochs=10000
)

### Initialization Options

- random: random non-negative initialization
- eye: identity-based initialization
- ssvd: non-negative SVD initialization (optional)

### Output Modes

- matrix: returns only factor matrices
- lists: returns only metric logs
- all: returns matrices and metrics (recommended)

## Metrics Tracked

### Reconstruction Error

- errorsGD: relative Frobenius error ||A - WH||² / ||A||²
- fullerrorsGD: absolute Frobenius error

### Rank Analysis

- rankGD: effective rank
- nuclearrankGD: nuclear norm / operator norm ratio

### Singular Values

- SVGD1: largest singular value
- SVGD2: second largest singular value

### Interpretation

- Effective rank plateau: good early stopping point (low-rank bias)
- Error stagnation: convergence
- Decreasing singular values: information compression

## Studying Low-Rank Bias

Low-rank bias refers to the tendency of models to learn low-complexity structures before fine details.

How to observe it:
1. Track effective rank during training
2. Detect the plateau
3. Apply early stopping
4. Visualize H signatures to see global structures first

## Visualizations

- Matrix heatmaps
- Training curves
- NMF signatures
- Reconstruction comparisons with error maps

## Contributors

- Thomas Lambelin
- Malo David
- Maxime Chansat
