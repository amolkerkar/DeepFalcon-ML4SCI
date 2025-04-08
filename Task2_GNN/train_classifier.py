'''
Trains the GCN on quark/gluon classification
'''


import torch
from torch_geometric.loader import DataLoader
from Task2_GNN.jet_graph_dataset import JetGraphDataset
from Task2_GNN.gnn_model import JetGCN
from tqdm import tqdm

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = JetGraphDataset(".\\data\\quark-gluon_data-set_n139306.hdf5", k=10, limit=5000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = JetGCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        total_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/10")
        for batch in loop:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} - Loss: {total_loss:.2f}")
