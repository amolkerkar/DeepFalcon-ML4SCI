import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

#center crop + hdf5 loading
class InMemoryJetDataset(Dataset):
    def __init__(self, file_path, limit=5000):
        #only loading the first N samples to avoid memory explosion in my PC
        with h5py.File(file_path, 'r') as f:
            data = f['X_jets'][:limit]
        data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        transform = T.CenterCrop((120, 120))
        self.data = torch.stack([transform(img) for img in data])


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx],  #returning tuple for DataLoader


#Simple VAE
class VAE(nn.Module):
    def __init__(self, latent_dim=32):  #compact latent space
        super(VAE, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),  #(16, 60, 60)
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), #(32, 30, 30)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), #(64, 15, 15)
            nn.ReLU(),
            nn.Flatten()
        )
        #for me to figure out conv output shape
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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = self.fc_decode(z)
        return self.dec(h.view(-1, *self._conv_shape))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


#VAE Loss
def vae_loss(x, x_hat, mu, logvar):
    recon = F.mse_loss(x_hat, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl

#Saving Recon Images
def save_reconstructions(model, loader, epoch, out_dir="outputs_debug"):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        imgs = batch[0][:5].to(device)
        recon, _, _ = model(imgs)

    fig, axes = plt.subplots(5, 2, figsize=(6, 12))
    for i in range(5):
        axes[i, 0].imshow(imgs[i].permute(1, 2, 0).cpu(), vmin=0, vmax=0.1)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(recon[i].permute(1, 2, 0).cpu(), vmin=0, vmax=0.1)
        axes[i, 1].set_title("Reconstruction")
        axes[i, 1].axis("off")

        print(f"[Image {i}] Pixel range: {imgs[i].min().item():.4f} to {imgs[i].max().item():.4f}")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/recon_epoch_{epoch+1}.png")
    plt.close()

#Training
def train_vae(model, loader, epochs=2, lr=1e-3, limit_batches=50):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        start = time.time()

        loop = tqdm(enumerate(loader), total=limit_batches, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (x,) in loop:
            if batch_idx >= limit_batches:
                break
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss = vae_loss(x, x_hat, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"[Epoch {epoch+1}] Total loss: {total_loss:.2f}, Time: {time.time() - start:.1f}s")
        save_reconstructions(model, loader, epoch)


#SSIM and PSNR calculations
def evaluate_recon_quality(originals, reconstructions):
    ssim_scores = []
    psnr_scores = []

    for i in range(len(originals)):
        orig = originals[i].mean(dim=0).cpu().numpy()
        recon = reconstructions[i].mean(dim=0).cpu().numpy()

        ssim_scores.append(ssim(orig, recon, data_range=1.0))
        psnr_scores.append(psnr(orig, recon, data_range=1.0))

    print(f"SSIM: mean={np.mean(ssim_scores):.4f}, std={np.std(ssim_scores):.4f}")
    print(f"PSNR: mean={np.mean(psnr_scores):.2f} dB, std={np.std(psnr_scores):.2f} dB")




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = InMemoryJetDataset(".\\data\\quark-gluon_data-set_n139306.hdf5", limit=5000) #setting according to my system limitations
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

    vae = VAE(latent_dim=32).to(device)

    train_vae(vae, loader, epochs=2, limit_batches=500)  #setting according to my system limitations
    torch.save(vae.state_dict(), ".\\models\\vae_jet_trained.pth")
    #model saved!

    vae.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        imgs = batch[0][:5].to(device)
        recon, _, _ = vae(imgs)

    evaluate_recon_quality(imgs, recon)
