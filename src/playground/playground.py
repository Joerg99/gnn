from helper import visualize_graph, engineer_features, to_pyg_data
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import HeteroData

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























