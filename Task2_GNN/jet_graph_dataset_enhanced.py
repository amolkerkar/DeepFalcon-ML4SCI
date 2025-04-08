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
            self.images = f['X_jets'][:limit]  # shape (N, 125, 125, 3)
            self.labels = f['y'][:limit]       # shape (N,)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # (125, 125, 3)
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        points = []
        for c in range(3):
            channel = img[:, :, c]
            nonzero = np.nonzero(channel)
            intensities = channel[nonzero]
            coords = np.stack(nonzero, axis=-1).astype(np.float32)
            coords /= 125.0  # normalize to [0, 1]
            dist = np.linalg.norm(coords - 0.5, axis=1, keepdims=True)
            feats = np.hstack([
                coords,                       # x, y (normalized)
                np.full((len(coords), 1), c), # channel
                intensities[:, None],         # intensity
                dist                          # distance to center
            ])
            points.append(feats)

        if len(points) == 0 or all(len(p) == 0 for p in points):
            return self.__getitem__((idx + 1) % len(self))

        points = np.concatenate(points, axis=0)
        pos = torch.tensor(points[:, :2], dtype=torch.float32)
        x = torch.tensor(points, dtype=torch.float32)  # node features before adding degree

        if len(pos) < self.k:
            return self.__getitem__((idx + 1) % len(self))

        nbrs = NearestNeighbors(n_neighbors=self.k).fit(pos)
        edge_index = nbrs.kneighbors_graph(pos, mode='connectivity').tocoo()
        edge_index = torch.tensor(np.vstack((edge_index.row, edge_index.col)), dtype=torch.long)

        # Degree feature
        row, _ = edge_index
        deg = torch.bincount(row, minlength=x.shape[0]).float().unsqueeze(1)
        x = torch.cat([x, deg], dim=1)  # final node features

        data = Data(x=x, edge_index=edge_index, y=y)
        return data
