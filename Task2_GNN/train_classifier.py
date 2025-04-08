'''
Trains the GCN on quark/gluon classification
'''


import torch
from sklearn.metrics import accuracy_score
from torch_geometric.loader import DataLoader
from jet_graph_dataset import JetGraphDataset
#from gnn_model import JetGCN
from gnn_model_graphsage import JetGraphSAGE
from tqdm import tqdm

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = JetGraphDataset(".\\data\\quark-gluon_data-set_n139306.hdf5", k=5, limit=3500)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = JetGraphSAGE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        total_loss = 0
        all_preds = []
        all_labels = []

        loop = tqdm(loader, desc=f"Epoch {epoch+1}/10")
        for batch in loop:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

            loop.set_postfix(loss=loss.item())

        # Compute accuracy at the end of the epoch
        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1} - Loss: {total_loss:.2f}, Accuracy: {acc:.4f}")
