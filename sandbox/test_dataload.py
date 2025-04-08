import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

def load_hdf5_dataset(file_path):
    with h5py.File(file_path, 'r') as f:
        images = np.array(f['X_jets'])

    #PyTorch format: (N, 3, 125, 125)
    images = np.transpose(images, (0, 3, 1, 2))
    images = images.astype(np.float32)
    images /= images.max()
    
    return torch.tensor(images)


data = load_hdf5_dataset(".\\data\\quark-gluon_data-set_n139306.hdf5")
print(data.shape)  #should be: (139306, 3, 125, 125) although cant load
