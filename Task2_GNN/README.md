# Task 2 – Jets as Graphs (GNN Classifier)

## What I did

This task was about representing jet events (originally in image format) as graphs and training a Graph Neural Network (GNN) to classify them as either **quark** or **gluon** jets.

To do this, I converted the calorimeter-style image data into a point cloud per event and then formed graphs using k-NN. These were then fed into a GCN for binary classification.


## Dataset details

File: `quark-gluon_data-set_n139306.hdf5`
- Shape of input: `(N, 125, 125, 3)` → 3 channels: `ECAL`, `HCAL`, and `Tracks`
- Labels: 0 = quark, 1 = gluon

## Graph construction pipeline

Defined in `jett_graph_dataset.py`
- For each event:
  - I took non-zero pixels from each channel
  - Stored the `(x, y)` location, channel index, and intensity as a 4D feature vector
  - Used `sklearn`’s `NearestNeighbors` to build k-NN edges (k=10)
  - Output is a `torch_geometric.data.Data` object with `x`, `edge_index`, and `y`

If any graph didn’t have enough points (less than `k`), I skipped to the next one to avoid degenerate graphs.

## GNN model

File: `gnn_model.py`

Model used:
```python
GCNConv(4 → 64) → ReLU → GCNConv(64 → 64) → ReLU → global_mean_pool → Linear → logits (2 classes)
