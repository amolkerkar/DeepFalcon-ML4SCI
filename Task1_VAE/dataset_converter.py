'''
This script was originally created to convert the large HDF5 jet dataset into smaller PyTorch `.pt` chunks 
for lazy loading due to memory limitations on my system. However, I later opted to continue using the HDF5 file directly 
because I was able to limit the dataset to the first 5000 samples (`limit=5000`) and load them fully into 
memory without issue. This allowed simpler data handling for training and evaluation during Task 1.
'''

import h5py
import torch
import os

def convert_to_smaller_chunks(hdf5_path, output_dir="jet_chunks", chunk_size=2000):
    os.makedirs(output_dir, exist_ok=True)
    with h5py.File(hdf5_path, 'r') as f:
        total_samples = f['X_jets'].shape[0]
        print("Total samples:", total_samples)

        for i in range(0, total_samples, chunk_size):
            print(f"Processing samples {i} to {min(i + chunk_size, total_samples)}")
            chunk = f['X_jets'][i:i + chunk_size]
            tensor = torch.tensor(chunk, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
            torch.save(tensor, os.path.join(output_dir, f"chunk_{i//chunk_size}.pt"))
            print(f"Saved: chunk_{i//chunk_size}.pt")

convert_to_smaller_chunks("quark-gluon_data-set_n139306.hdf5")
