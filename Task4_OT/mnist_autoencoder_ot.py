'''
Implemented a basic MLP-based autoencoder with a **2D latent space**.
- Trained it on **MNIST digits 0 and 4**.
- Used the **POT (Python Optimal Transport)** library to:
  - Map latent vectors to a Gaussian using **OT**.
  - Decode and visualize results.
  - Also decoded from raw Gaussian samples as a bonus comparison
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import ot  # POT library

# ---------------- Model ----------------
class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

#Training
def train(model, loader, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = x.to(device)
            x_hat, _ = model(x)
            loss = F.mse_loss(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

#Visualse
def show_images(imgs, title):
    fig, axes = plt.subplots(1, len(imgs), figsize=(12, 2))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i].squeeze(), cmap='gray')
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#Loading MNIST and filter digits 0 and 4
transform = transforms.ToTensor()
dataset = datasets.MNIST(root=".", train=True, download=True, transform=transform)
filtered_indices = [i for i, (_, label) in enumerate(dataset) if label in [0, 4]]
filtered_dataset = Subset(dataset, filtered_indices)
loader = DataLoader(filtered_dataset, batch_size=128, shuffle=True)

model = AutoEncoder(latent_dim=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train(model, loader, optimizer, epochs=20)

#Geting latent representations
model.eval()
latents, images = [], []
with torch.no_grad():
    for x, _ in DataLoader(filtered_dataset, batch_size=512):
        x = x.to(device)
        _, z = model(x)
        latents.append(z.cpu())
        images.append(x.cpu())
latents = torch.cat(latents)
images = torch.cat(images)

#Performing the optimal transport mapping to N(0,1)
target = torch.randn_like(latents)
M = ot.dist(latents.numpy(), target.numpy(), metric='euclidean')
M /= M.max()
T = ot.emd(np.ones(len(latents)) / len(latents), np.ones(len(target)) / len(target), M)
z_ot = torch.tensor(T @ target.numpy(), dtype=torch.float32)

#Normalize
scaler = StandardScaler()
z_ot_scaled = torch.tensor(scaler.fit_transform(z_ot), dtype=torch.float32)

#Decoding from OTmapped latent space
with torch.no_grad():
    decoded = model.decoder(z_ot_scaled.to(device))[:8].cpu()

show_images(decoded, "OT mapped decoded")
mapped = ot.da.MappingTransport(kernel="linear", max_iter=2000)
