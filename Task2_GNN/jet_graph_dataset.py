'''
converts 125x125x3 jet images to point clouds and graphs
'''

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors

class JetGraphDataset(Dataset):
    def __init__(self, h5_path, k=10, limit=5000):
        super().__init__()
        self.k = k

        with h5py.File(h5_path, 'r') as f:
            self.images = f['X_jets'][:limit]   #shape(N, 125, 125, 3)
            self.labels = f['y'][:limit]       #shape(N,)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  #(125, 125, 3)
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)


        points = []
        for c in range(3):
            channel = img[:, :, c]
            nonzero = np.nonzero(channel)
            intensities = channel[nonzero]
            coords = np.stack(nonzero, axis=-1)
            feats = np.hstack([coords, np.full((len(coords), 1), c), intensities[:, None]])
            points.append(feats)

        if len(points) == 0 or all(len(p) == 0 for p in points):
            return self.__getitem__((idx + 1) % len(self))  #fallback

        points = np.concatenate(points, axis=0)
        pos = torch.tensor(points[:, :2], dtype=torch.float32)
        x = torch.tensor(points, dtype=torch.float32)  #full 4D feature

        if len(pos) < self.k:
            return self.__getitem__((idx + 1) % len(self))

        nbrs = NearestNeighbors(n_neighbors=self.k).fit(pos)
        edge_index = nbrs.kneighbors_graph(pos, mode='connectivity').tocoo()
        edge_index = torch.tensor(np.vstack((edge_index.row, edge_index.col)), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        return data