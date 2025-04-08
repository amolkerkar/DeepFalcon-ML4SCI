# Task 4 - Autoencoder + Optimal Transport (MNIST & Jet Images)

This part of the project was about experimenting with **optimal transport (OT)** applied to the latent space of an autoencoder.

---

## What the Task Asked

- Train an autoencoder on 2 MNIST digits of choice (I picked **0** and **4**).
- Map the learned latent space to a standard normal distribution using OT and decode from it.
- (Bonus) Sample directly from a Gaussian and decode.
- Apply the same process to the **jet dataset** used in Task 1.

---

## What I Did

### `mnist_autoencoder_ot.py`
- Implemented a basic MLP-based autoencoder with a **2D latent space**.
- Trained it on **MNIST digits 0 and 4**.
- Used the **POT (Python Optimal Transport)** library to:
  - Map latent vectors to a Gaussian using **OT**.
  - Decode and visualize results.
  - Also decoded from raw Gaussian samples as a bonus comparison.

### `jet_ot_decoder.py`
- Used the trained VAE from Task 1 (`vae_jet_trained.pth`) on the **jet image dataset**.
- Extracted the latent `mu` vectors.
- Mapped the latent space to a Gaussian using OT.
- Decoded both:
  - OT-mapped latent vectors (produces more realistic samples)
  - Random Gaussian samples (gives more variety)



## Output

- All visual outputs are saved to the `outputs_jet/` directory.
  - **OT-mapped decoded jets** → `ot_decoded.png`
  - **Gaussian sampled decoded jets** → `gaussian_decoded.png`

These results show that OT-mapped samples tend to look more realistic and smoother than directly sampling from a Gaussian.



## Why this is good

The OT mapping ensures that latent vectors sampled from a Gaussian are transported closer to the actual **data manifold**, leading to better and more meaningful generations — especially useful in real-world datasets like jet images where the latent space is far from ideal.


