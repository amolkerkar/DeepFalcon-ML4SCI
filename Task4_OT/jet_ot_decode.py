'''
Used the trained VAE from Task 1 (`vae_jet_trained.pth`) on the **jet image dataset**.
- Extracted the latent `mu` vectors.
- Mapped the latent space to a Gaussian using OT.
- Decoded both:
  - OT-mapped latent vectors (produces more realistic samples)
  - Random Gaussian samples (gives more variety)
'''


import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import h5py
import ot
from tqdm import tqdm

class InMemoryJetDataset(Dataset):
    def __init__(self, file_path, limit=5000):
        with h5py.File(file_path, 'r') as f:
            data = f['X_jets'][:limit]
        data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        transform = T.CenterCrop((120, 120))
        self.data = torch.stack([transform(img) for img in data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx],

#VAE
class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        dummy_input = torch.zeros(1, 3, 120, 120)
        dummy_output = self.enc[:-1](dummy_input)
        self._conv_shape = dummy_output.shape[1:]
        self._flattened_dim = int(np.prod(self._conv_shape))

        self.fc_mu = nn.Linear(self._flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._flattened_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self._flattened_dim)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.fc_decode(z)
        return self.dec(h.view(-1, *self._conv_shape))

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decode(z), mu, logvar

#Latent Extraction
def get_latents(model, loader):
    model.eval()
    latents = []
    with torch.no_grad():
        for x, in loader:
            x = x.to(device)
            mu, _ = model.encode(x)
            latents.append(mu.cpu())
    return torch.cat(latents)

#OT Mapping
def ot_map_to_normal(source_latents, n_samples=1000):
    source = source_latents.numpy()
    target = np.random.randn(n_samples, source.shape[1])
    a = np.ones((len(source),)) / len(source)
    b = np.ones((len(target),)) / len(target)
    ot_plan = ot.emd(a, b, ot.dist(source, target))
    mapped = ot_plan @ target
    return torch.tensor(mapped, dtype=torch.float32)

#Decode and Visualization
def decode_samples(model, z_samples, title="Decoded", output_path="Task4_OT/outputs_jet/decoded.png"):
    model.eval()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with torch.no_grad():
        decoded = model.decode(z_samples.to(device)).cpu()

    fig, axes = plt.subplots(1, 8, figsize=(16, 2))
    for i in range(8):
        axes[i].imshow(decoded[i].permute(1, 2, 0), vmin=0, vmax=0.1)
        axes[i].axis('off')
    plt.suptitle(title)
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = InMemoryJetDataset(".\\data\\quark-gluon_data-set_n139306.hdf5", limit=5000)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    vae = VAE(latent_dim=32).to(device)
    vae.load_state_dict(torch.load(".\\models\\vae_jet_trained.pth", map_location=device))
    vae.eval()

    # Get encoded latent space from real samples
    latents = get_latents(vae, loader)

    # OT-mapped latent vectors
    z_mapped = ot_map_to_normal(latents, n_samples=64)
    decode_samples(vae, z_mapped, title="OT mapped decoded (jets)", output_path="outputs_jet/ot_decoded.png")

    # Gaussian-sampled latent vectors
    z_random = torch.randn_like(z_mapped).to(device)
    decode_samples(vae, z_random, title="Gaussian sampled decoded (jets)", output_path="outputs_jet/gaussian_decoded.png")
