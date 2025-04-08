## Task 1 - VAE for Jet Images (DeepFalcon GSoC 2025)

This task focuses on building and training a Variational Autoencoder (VAE) for quark-gluon jet images. The dataset used is provided as an HDF5 file. The ultimate goal was to reconstruct the input images with high fidelity and evaluate the model performance both visually and through image similarity metrics like SSIM and PSNR.

### Dataset
- Source: `quark-gluon_data-set_n139306.hdf5`
- Input shape: (3, 125, 125), center-cropped to (120, 120)
- Initially, I wrote a utility (dataset_convertor.py) to convert the HDF5 into smaller .pt files due to memory limitations, but later found it more efficient to load the HDF5 lazily in chunks using PyTorch Dataset and just limit the number of samples during training.

## Model Architecture

A fairly simple convolutional VAE was implemented. It consists of:
- * Encoder: Conv2D layers with increasing channels (3 → 64)
- ReLU activations
- Flattened and passed through 2 linear layers for mean and log-variance of the latent space
- Latent space: Dimension = 32 Sampled via the reparameterization trick

- * Decoder:
- Fully connected layer to reshape the latent vector back to conv feature map size
- ConvTranspose2D layers to upscale back to image dimensions (3, 120, 120)
- Final activation: Sigmoid (since image pixels are in [0, 1])

Loss Function
- Total Loss = Reconstruction Loss + KL Divergence
- Reconstruction: Mean Squared Error (MSE)
- KL Divergence: Standard VAE KL term between latent distribution and standard normal

## Output Visualization

Every epoch, 5 random samples from the batch are reconstructed and saved as a side-by-side image grid:
![Reconstruction Example](Task1_VAE/outputs_debug/recon_epoch_15.png)


## Evaluation Metrics

At the end of training, I computed the following metrics on a mini-batch of 5 samples:
SSIM (Structural Similarity Index)
PSNR (Peak Signal-to-Noise Ratio)
These metrics give a more quantitative idea of the perceptual quality of reconstructions.

## Final Scores (after 15 epochs):

SSIM: mean=1.0000, std=0.0000
PSNR: mean=106.32 dB, std=2.93 dB

## My Observations

The VAE quickly learned to reconstruct jet images within a few epochs (sub-25 seconds per epoch).
Image pixel intensities are very low (~0.001), which is expected from pre-processed calorimeter images.
The decoder does a decent job reproducing sparse energy deposits, despite such low magnitude data.

## Training Log Snippet

Epoch 2/15: 100%|██████████████████████████████████████████████████| 1500/1500 [00:22<00:00, 67.29it/s, loss=2.43]
[Epoch 2] Total loss: 128411.83, Time: 22.3s
...
Epoch 15/15: 100%|█████████████████████████████████████████████████| 1500/1500 [00:24<00:00, 60.85it/s, loss=0.00015]

SSIM: mean=1.0000, std=0.0000
PSNR: mean=106.32 dB, std=2.93 dB
![alt text](assets/task1_op.png)

## Files:

vae_jet_trainer.py: Full code for dataset, VAE, training loop, and evaluation
dataset_convertor.py: Optional utility for slicing HDF5 into .pt chunks
outputs_debug/: Folder with reconstruction images per epoch
models/vae_jet_trained.pth: model saved!