from torch.utils.data import Dataset
import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset

class JetImageDataset(Dataset):
    def __init__(self, file_path):
        self.file = h5py.File(file_path, 'r')
        self.dataset = self.file['X_jets']  #shape(N, 125, 125, 3)
        self.length = self.dataset.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.dataset[idx]  #shape(125, 125, 3)
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # to(3, 125, 125)
        img /= 255.0
        return img,

    def close(self):
        self.file.close()

file_path = ".\\data\\quark-gluon_data-set_n139306.hdf5"
dataset = JetImageDataset(file_path)
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2) #keeping workers 0 is better sometiesm
