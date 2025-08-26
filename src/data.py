import torch
from torch_geometric.data import Data

# Node features: [num_nodes, num_features]
# Let's say we have 2 features per person: age and number of hobbies
x = torch.tensor([
    [25, 2],  # Node 0
    [30, 5],  # Node 1
    [22, 1],  # Node 2
    [40, 6]   # Node 3
], dtype=torch.float)

# Edge index: [2, num_edges]
# Connections: 0--1, 0--2, 1--2, 1--3
# Remember, graphs are often undirected, so we define edges in both directions.
edge_index = torch.tensor([
    [0, 0, 1, 1, 2, 3],  # Source nodes
    [1, 2, 0, 2, 0, 1]   # Target nodes
], dtype=torch.long)

# Let's say we want to predict if a person is likely to be a "connector" (label 1) or not (label 0)
y = torch.tensor([1, 1, 0, 0], dtype=torch.long)

# Create the PyG Data object
graph_data = Data(x=x, edge_index=edge_index, y=y)

print(graph_data)
# >>> Data(x=[4, 2], edge_index=[2, 6], y=[4])