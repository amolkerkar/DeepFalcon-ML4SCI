'''
Trains the GCN on quark/gluon classification
plots roc and evaluated too!
'''


import torch
from torch_geometric.loader import DataLoader
from Task2_GNN.jet_graph_dataset_enhanced_2 import JetGraphDatasetEnhanced, JetEdgeConvNet


from gnn_model_gat import JetGAT
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = JetGraphDatasetEnhanced("data/quark-gluon_data-set_n139306.hdf5", k=10, limit=5000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = JetEdgeConvNet().to(device)
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


    #Evaluation & ROC Curve below
   
    print("\nEvaluating ROC curve...")
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of class 1 (e.g. gluon)
            all_probs.append(probs.cpu())
            all_labels.append(batch.y.cpu())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    # Plotting
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Jet Classification ROC Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("Task2_GNN/roc_curve.png")
    plt.show()
