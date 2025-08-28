from helper import visualize_graph, engineer_features, to_pyg_data
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import HeteroData
import torch.nn.functional as F
from torch.optim import Adam

from torch_geometric.nn import GATConv, to_hetero
from torch_geometric.loader import LinkNeighborLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

from datetime import datetime
import matplotlib.pyplot as plt

# Create a list of dictionaries, where each dictionary represents an edge and its context.
# This format is easy to read and convert to a DataFrame.
data = [
    # Customer -> Region Edges
    {'source': 'C1', 'source_type': 'customer', 'age': 28, 'gender': 'M',
     'target': 'R1', 'target_type': 'region', 'region_name': 'North America',
     'edge_type': 'lives_in'},
    {'source': 'C2', 'source_type': 'customer', 'age': 45, 'gender': 'F',
     'target': 'R2', 'target_type': 'region', 'region_name': 'Europe',
     'edge_type': 'lives_in'},
    {'source': 'C3', 'source_type': 'customer', 'age': 33, 'gender': 'F',
     'target': 'R2', 'target_type': 'region', 'region_name': 'Europe',
     'edge_type': 'lives_in'},
    {'source': 'C4', 'source_type': 'customer', 'age': 19, 'gender': 'M',
     'target': 'R1', 'target_type': 'region', 'region_name': 'North America',
     'edge_type': 'lives_in'},

    # Customer -> Product "buys" Edges
    {'source': 'C1', 'source_type': 'customer', 'age': 28, 'gender': 'M',
     'target': 'P101', 'target_type': 'product', 'product_name': 'Laptop',
     'edge_type': 'buys', 'quantity': 1, 'price': 1200},
    {'source': 'C1', 'source_type': 'customer', 'age': 28, 'gender': 'M',
     'target': 'P205', 'target_type': 'product', 'product_name': 'Mouse',
     'edge_type': 'buys', 'quantity': 1, 'price': 75},
    {'source': 'C2', 'source_type': 'customer', 'age': 45, 'gender': 'F',
     'target': 'P450', 'target_type': 'product', 'product_name': 'Webcam',
     'edge_type': 'buys', 'quantity': 2, 'price': 90},
    {'source': 'C3', 'source_type': 'customer', 'age': 33, 'gender': 'F',
     'target': 'P101', 'target_type': 'product', 'product_name': 'Laptop',
     'edge_type': 'buys', 'quantity': 1, 'price': 1150},
    {'source': 'C3', 'source_type': 'customer', 'age': 33, 'gender': 'F',
     'target': 'P300', 'target_type': 'product', 'product_name': 'Keyboard',
     'edge_type': 'buys', 'quantity': 1, 'price': 110},
    {'source': 'C4', 'source_type': 'customer', 'age': 19, 'gender': 'M',
     'target': 'P205', 'target_type': 'product', 'product_name': 'Mouse',
     'edge_type': 'buys', 'quantity': 1, 'price': 80},

    # Customer -> Product "viewed" Edges
    {'source': 'C1', 'source_type': 'customer', 'age': 28, 'gender': 'M',
     'target': 'P300', 'target_type': 'product', 'product_name': 'Keyboard',
     'edge_type': 'viewed', 'duration_seconds': 180},
    {'source': 'C2', 'source_type': 'customer', 'age': 45, 'gender': 'F',
     'target': 'P101', 'target_type': 'product', 'product_name': 'Laptop',
     'edge_type': 'viewed', 'duration_seconds': 300},
    {'source': 'C4', 'source_type': 'customer', 'age': 19, 'gender': 'M',
     'target': 'P450', 'target_type': 'product', 'product_name': 'Webcam',
     'edge_type': 'viewed', 'duration_seconds': 45},
]

# Create the DataFrame
df = pd.DataFrame(data)

print("--- Unified Pandas DataFrame ---")
print(df)

# Create an empty directed graph
G = nx.DiGraph()

# Iterate over each row in the DataFrame to build the graph
for index, row in df.iterrows():
    # --- Add nodes with their attributes ---
    # We use a check to ensure we only add each node once, preventing attribute overwrites.

    # Add source node if it doesn't exist
    if not G.has_node(row['source']):
        if row['source_type'] == 'customer':
            G.add_node(row['source'],
                       node_type='customer',
                       age=row['age'],
                       gender=row['gender'])

    # Add target node if it doesn't exist
    if not G.has_node(row['target']):
        if row['target_type'] == 'region':
            G.add_node(row['target'],
                       node_type='region',
                       name=row['region_name'])
        elif row['target_type'] == 'product':
            G.add_node(row['target'],
                       node_type='product',
                       name=row['product_name'])

    # --- Add the edge with its attributes ---
    edge_attrs = {'edge_type': row['edge_type']}
    if pd.notna(row.get('price')):
        edge_attrs['price'] = row['price']
    if pd.notna(row.get('quantity')):
        edge_attrs['quantity'] = row['quantity']
    if pd.notna(row.get('duration_seconds')):
        edge_attrs['duration'] = row['duration_seconds']

    G.add_edge(row['source'], row['target'], **edge_attrs)

print("\n--- NetworkX Graph Created ---")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print("\nSample Node Data:")
print(list(G.nodes(data=True))[:3])
print("\nSample Edge Data:")
print(list(G.edges(data=True))[:3])

# visualize_graph(G)


# Execute the function
processed_data = engineer_features(df)

data = to_pyg_data(processed_data)

data

from torch_geometric.transforms import ToUndirected

data = ToUndirected()(data)



class GAT_GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False, heads=2)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


# This class wraps the GNN and adds the link prediction head
class Model(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata):
        super().__init__()
        # The base GNN model
        self.gnn = GAT_GNN(hidden_channels, out_channels)
        # Wrap it with to_hetero to make it work on our graph
        self.hetero_gnn = to_hetero(self.gnn, metadata, aggr='sum')

    def forward(self, data):
        # The forward pass returns the embeddings for all nodes
        return self.hetero_gnn(data.x_dict, data.edge_index_dict)

    def predict_links(self, embeddings, edge_label_index):
        # Predict links for the ('customer', 'buys', 'product') edge type
        src_emb = embeddings['customer'][edge_label_index[0]]
        dst_emb = embeddings['product'][edge_label_index[1]]
        # Dot product as a simple similarity measure
        score = (src_emb * dst_emb).sum(dim=-1)
        return score


# --- Step 3: Training Setup ---

# Instantiate the model
EMBEDDING_SIZE = 64
model = Model(hidden_channels=128, out_channels=EMBEDDING_SIZE, metadata=data.metadata())

# Define the training task using LinkNeighborLoader
# We want to predict 'buys' relationships
train_loader = LinkNeighborLoader(
    data=data,
    num_neighbors=[10, 5],  # Sample 10 neighbors at 1st hop, 5 at 2nd hop
    batch_size=32,
    # Specify the edge type for link prediction
    edge_label_index=(('customer', 'buys', 'product'), data['customer', 'buys', 'product'].edge_index),
    # Generate an equal number of negative samples
    neg_sampling_ratio=1.0
)

# Define optimizer and loss function
optimizer = Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCEWithLogitsLoss()


# --- Step 4: Training Loop ---
def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        # Get embeddings for the nodes in the current mini-batch
        embeddings = model(batch)
        # Predict the links we are interested in
        preds = model.predict_links(embeddings, batch['customer', 'buys', 'product'].edge_label_index)
        # Get the ground truth labels (1 for positive edges, 0 for negative)
        ground_truth = batch['customer', 'buys', 'product'].edge_label

        loss = loss_fn(preds, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * preds.numel()
    return total_loss / len(train_loader.dataset)

# pip install pyg-lib -f https://data.pyg.org/whl/torch-${'1.12.0'}+${'cpu'}.html
# pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cpu.html


print("Starting model training...")
for epoch in range(1, 51):
    loss = train()
    if epoch % 10 == 0:
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")


# --- Step 5: Inference and Finding Similar Customers ---

@torch.no_grad()
def get_customer_embeddings():
    model.eval()
    # Get the final embeddings for all nodes in the graph
    all_embeddings = model(data)
    # Return just the customer embeddings
    return all_embeddings['customer']


# Get the learned embeddings
customer_embeddings = get_customer_embeddings()
print(f"\nLearned Customer Embeddings Shape: {customer_embeddings.shape}")

# Let's find customers similar to C1
customer_encoder = processed_data['encoders']['customer']
target_customer_name = 'C1'
target_customer_idx = customer_encoder.transform([target_customer_name])[0]

# Calculate cosine similarity between the target customer and all other customers
similarities = cosine_similarity(
    customer_embeddings[target_customer_idx].reshape(1, -1),
    customer_embeddings
)

# Get the indices of customers sorted by similarity (most similar first)
# np.argsort returns indices, [::-1] reverses them for descending order
most_similar_indices = np.argsort(similarities[0])[::-1]

print(f"\n--- Most Similar Customers to {target_customer_name} ---")
for idx in most_similar_indices:
    customer_name = customer_encoder.inverse_transform([idx])[0]
    similarity_score = similarities[0][idx]
    if customer_name != target_customer_name:
        print(f"Customer: {customer_name}, Similarity Score: {similarity_score:.4f}")




















