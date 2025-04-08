import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from math import atan2

class JetGraphDatasetEnhanced(Dataset):
    def __init__(self, h5_path, k=10, limit=5000):
        self.k = k
        with h5py.File(h5_path, 'r') as f:
            self.images = f['X_jets'][:limit]
            self.labels = f['y'][:limit]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        points = []
        for c in range(3):
            channel = img[:, :, c]
            nonzero = np.nonzero(channel)
            intensities = channel[nonzero]
            coords = np.stack(nonzero, axis=-1)
            r = np.linalg.norm(coords, axis=1, keepdims=True)
            theta = np.arctan2(coords[:, 1], coords[:, 0]).reshape(-1, 1)
            feats = np.hstack([
                coords,                         # i, j
                np.full((len(coords), 1), c),   # channel
                intensities[:, None],           # intensity
                r,                              # radial distance
                theta                           # angle
            ])
            points.append(feats)

        if len(points) == 0 or all(len(p) == 0 for p in points):
            return self.__getitem__((idx + 1) % len(self))

        points = np.concatenate(points, axis=0)
        pos = torch.tensor(points[:, :2], dtype=torch.float32)
        x = torch.tensor(points, dtype=torch.float32)

        if len(pos) < self.k:
            return self.__getitem__((idx + 1) % len(self))

        nbrs = NearestNeighbors(n_neighbors=self.k).fit(pos)
        edge_index = nbrs.kneighbors_graph(pos, mode='connectivity').tocoo()
        edge_index = torch.tensor(np.vstack((edge_index.row, edge_index.col)), dtype=torch.long)

        degree = torch.bincount(edge_index[0], minlength=pos.shape[0]).float().unsqueeze(1)
        x = torch.cat([x, degree], dim=1)  # add degree as final feature

        return Data(x=x, edge_index=edge_index, y=y)


# EdgeConv model using MLP
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_mean_pool

class JetEdgeConvNet(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64):
        super().__init__()
        self.edge_conv1 = EdgeConv(nn=nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))
        self.edge_conv2 = EdgeConv(nn=nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.edge_conv1(x, edge_index)
        x = self.edge_conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.classifier(x)
