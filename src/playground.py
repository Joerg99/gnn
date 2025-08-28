import networkx as nx
import torch
from torch_geometric.data import HeteroData
import pandas as pd
from datetime import datetime

# --- 1. Create a Directed Graph in NetworkX ---
# A directed graph is better here as actions are directional (e.g., Customer BUYS Product)
G = nx.DiGraph()

# --- 2. Define and Add Nodes with Attributes ---
# We'll add a 'node_type' attribute to easily distinguish them later.

# Customer Nodes
customers = [
    {'id': 'C1', 'age': 28, 'gender': 'M'},
    {'id': 'C2', 'age': 45, 'gender': 'F'},
    {'id': 'C3', 'age': 33, 'gender': 'F'},
    {'id': 'C4', 'age': 19, 'gender': 'M'},
]
for c in customers:
    G.add_node(c['id'], node_type='customer', age=c['age'], gender=c['gender'])

# Region Nodes
regions = [
    {'id': 'R1', 'name': 'North America'},
    {'id': 'R2', 'name': 'Europe'},
]
for r in regions:
    G.add_node(r['id'], node_type='region', name=r['name'])

# Product Nodes
products = [
    {'id': 'P101', 'name': 'Laptop'},
    {'id': 'P205', 'name': 'Mouse'},
    {'id': 'P300', 'name': 'Keyboard'},
    {'id': 'P450', 'name': 'Webcam'},
]
for p in products:
    G.add_node(p['id'], node_type='product', name=p['name'])


# --- 3. Define and Add Edges with Meaningful Properties ---

# LIVES_IN relationships (Customer -> Region)
G.add_edge('C1', 'R1', edge_type='lives_in')
G.add_edge('C2', 'R2', edge_type='lives_in')
G.add_edge('C3', 'R2', edge_type='lives_in')
G.add_edge('C4', 'R1', edge_type='lives_in')

# BUYS relationships (Customer -> Product)
# Properties: date of purchase, quantity, price
G.add_edge('C1', 'P101', edge_type='buys', date=datetime(2023, 1, 15), quantity=1, price=1200)
G.add_edge('C1', 'P205', edge_type='buys', date=datetime(2023, 1, 15), quantity=1, price=75)
G.add_edge('C2', 'P450', edge_type='buys', date=datetime(2023, 2, 20), quantity=2, price=90)
G.add_edge('C3', 'P101', edge_type='buys', date=datetime(2023, 3, 5), quantity=1, price=1150)
G.add_edge('C3', 'P300', edge_type='buys', date=datetime(2023, 3, 5), quantity=1, price=110)
G.add_edge('C4', 'P205', edge_type='buys', date=datetime(2023, 4, 1), quantity=1, price=80)


# VIEWED relationships (Customer -> Product)
# Properties: timestamp of view, duration of view in seconds
G.add_edge('C1', 'P300', edge_type='viewed', timestamp=datetime(2023, 1, 12), duration_seconds=180)
G.add_edge('C2', 'P101', edge_type='viewed', timestamp=datetime(2023, 2, 18), duration_seconds=300)
G.add_edge('C4', 'P450', edge_type='viewed', timestamp=datetime(2023, 3, 28), duration_seconds=45)

print("--- NetworkX Graph Created ---")
print(f"Nodes: {G.nodes(data=True)}")
print(f"Edges: {G.edges(data=True)}")
print("-" * 30)

# --- 4. Prepare for PyTorch Geometric Conversion ---

# Create mappings from original node IDs to new integer indices
customer_mapping = {node_id: i for i, node_id in enumerate(c['id'] for c in customers)}
region_mapping = {node_id: i for i, node_id in enumerate(r['id'] for r in regions)}
product_mapping = {node_id: i for i, node_id in enumerate(p['id'] for p in products)}

# --- Node Feature Engineering ---

# Customer features: [age, gender_is_M, gender_is_F]
# We'll one-hot encode gender.
customer_features = []
for c in customers:
    gender_ohe = [1, 0] if c['gender'] == 'M' else [0, 1]
    customer_features.append([c['age']] + gender_ohe)
customer_x = torch.tensor(customer_features, dtype=torch.float)

# Region features: One-hot encoded region name
# Using pandas for easy one-hot encoding
region_df = pd.DataFrame(regions)
region_x_df = pd.get_dummies(region_df['name'], prefix='region')
region_x = torch.tensor(region_x_df.values, dtype=torch.float)

# Product features: One-hot encoded product name
product_df = pd.DataFrame(products)
product_x_df = pd.get_dummies(product_df['name'], prefix='product')
product_x = torch.tensor(product_x_df.values, dtype=torch.float)

# --- Edge and Edge Feature Engineering ---

# Initialize lists to store edge information
lives_in_edges = []
buys_edges, buys_attr = [], []
viewed_edges, viewed_attr = [], []

for src, dst, data in G.edges(data=True):
    edge_type = data['edge_type']

    if edge_type == 'lives_in':
        # Simple edge, no attributes to store
        src_idx = customer_mapping[src]
        dst_idx = region_mapping[dst]
        lives_in_edges.append([src_idx, dst_idx])

    elif edge_type == 'buys':
        src_idx = customer_mapping[src]
        dst_idx = product_mapping[dst]
        buys_edges.append([src_idx, dst_idx])

        # Edge features: [days_since_epoch, quantity, price]
        # Convert date to a numerical format
        days_since_epoch = (data['date'] - datetime(1970, 1, 1)).days
        buys_attr.append([days_since_epoch, data['quantity'], data['price']])

    elif edge_type == 'viewed':
        src_idx = customer_mapping[src]
        dst_idx = product_mapping[dst]
        viewed_edges.append([src_idx, dst_idx])

        # Edge features: [timestamp_seconds, duration_seconds]
        timestamp_seconds = (data['timestamp'] - datetime(1970, 1, 1)).total_seconds()
        viewed_attr.append([timestamp_seconds, data['duration_seconds']])

# --- 5. Create the HeteroData Object ---

data = HeteroData()

# Add node data
data['customer'].x = customer_x
data['region'].x = region_x
data['product'].x = product_x

# Add edge data
# PyG expects edge_index in shape [2, num_edges]
data['customer', 'lives_in', 'region'].edge_index = torch.tensor(lives_in_edges, dtype=torch.long).t().contiguous()
data['customer', 'buys', 'product'].edge_index = torch.tensor(buys_edges, dtype=torch.long).t().contiguous()
data['customer', 'buys', 'product'].edge_attr = torch.tensor(buys_attr, dtype=torch.float)
data['customer', 'viewed', 'product'].edge_index = torch.tensor(viewed_edges, dtype=torch.long).t().contiguous()
data['customer', 'viewed', 'product'].edge_attr = torch.tensor(viewed_attr, dtype=torch.float)

print("\n--- PyTorch Geometric HeteroData Object ---")
print(data)
print("-" * 30)

# You can also validate the graph structure
print("\nIs the graph valid?")
print(data.validate())







from pyvis.network import Network

# (Again, assume the NetworkX graph G has been created)

# 1. Create a pyvis Network instance
# notebook=True is essential for use in Jupyter/Colab.
net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white', notebook=True, directed=True)

# 2. Define color map (can be the same as before)
color_map = {
    'customer': '#007bff',  # A nice blue
    'region': '#28a745',  # A nice green
    'product': '#dc3545'  # A nice red
}

# 3. Add nodes to the pyvis network, adding attributes for hover-info
for node, attrs in G.nodes(data=True):
    # The 'title' attribute creates the hover-over tooltip
    title = f"Type: {attrs['node_type']}\n"
    title += "\n".join([f"{key}: {value}" for key, value in attrs.items() if key != 'node_type'])

    net.add_node(
        node,
        label=node,
        color=color_map[attrs['node_type']],
        title=title,
        size=25
    )

# 4. Add edges, with their own hover-info
for src, dst, attrs in G.edges(data=True):
    # Create a hover-over tooltip for the edge
    title = f"Type: {attrs['edge_type']}\n"
    title += "\n".join([f"{key}: {value}" for key, value in attrs.items() if key != 'edge_type'])

    net.add_edge(
        src,
        dst,
        title=title,
        label=attrs['edge_type']
    )

# 5. Add physics options for a better layout
net.show_buttons(filter_=['physics'])

# 6. Generate the HTML file (and display it in the notebook)
try:
    net.save_graph('c:/workspace/gnn/customer_graph.html')

except NameError:
    # If not in an interactive environment, just save the file
    print("Graph saved to customer_graph.html")