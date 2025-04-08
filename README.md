# ML4SCI GSoC 2025 – DeepFalcon Project (Amol Anil Kerkar)

This repository contains my submission for **GSoC 2025 under the ML4SCI DeepFalcon project**, implementing three key tasks based on Variational Autoencoders (VAEs), Graph Neural Networks (GNNs), and Optimal Transport (OT) on jet image data and MNIST.


## Folder Structure

| Folder         | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `Task1_VAE/`   | VAE-based autoencoder trained on jet images for reconstruction + metrics.  |
| `Task2_GNN/`   | Jet classification using a GNN where jets are converted into graphs.       |
| `Task4_OT/`    | Optimal Transport-based generation using a trained autoencoder (MNIST + Jets). |
| `data/`        | Place to store `quark-gluon_data-set_n139306.hdf5` and model weights.      |
| `models/`      | Trained model weights like `vae_jet_trained.pth` go here.                  |
| `sandbox/`     | Extra experimentation or in-progress scripts.                              |
| `requirements.txt` | Dependencies for all tasks combined.                                   |



## Task Descriptions & Usage

### 🔹 Task 1 – VAE Jet Reconstruction (`Task1_VAE/`)
**Goal:** Train a small convolutional VAE to reconstruct jet images and compute SSIM + PSNR.

-  Main file: `train_vae_jet.py`
-  Dataset: `data/quark-gluon_data-set_n139306.hdf5`
-  Output: Reconstructed images in `outputs_debug/`, model saved as `vae_jet_trained.pth`
-  Metrics: SSIM, PSNR after training.

### 🔹 Task 2 – GNN for Jet Classification (`Task2_GNN/`)
**Goal:** Treat each jet as a graph using k-nearest neighbors and classify using GCN.

-  Main file: `train_classifier.py`
-  Graph conversion: `jett_graph_dataset.py`
-  Model: `gnn_model.py`
-  Output: Loss logs per epoch, classification-ready GNN trained on edge graphs from image channels.

### 🔹 Task 4 – Optimal Transport on Latent Space (`Task4_OT/`)
**Goal:** Map latent space of autoencoder to a Gaussian distribution using Optimal Transport.

-  Jet OT pipeline: `ot_jet_pipeline.py` (uses saved VAE from Task 1)
-  MNIST example: `vae_ot_mnist.py`
-  Outputs:
  - `outputs_jet/ot_decoded.png`: OT-mapped decoded jet images.
  - `outputs_jet/gaussian_decoded.png`: Gaussian-sampled jet reconstructions.
  - MNIST decoded digits using OT shown via `matplotlib`.

---

## Requirements

Install all dependencies via:

```bash
pip install -r requirements.txt
