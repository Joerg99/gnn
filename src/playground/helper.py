from pyvis.network import Network
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch_geometric.data import HeteroData



def engineer_features(df: pd.DataFrame):
    """
    Processes a raw DataFrame of nodes and edges into numerical formats.

    Args:
        df: A DataFrame where each row is an edge, with node attributes included.

    Returns:
        A dictionary containing processed DataFrames and encoders for each entity.
    """

    # --- 1. Extract and Process Nodes ---

    # Customers
    customers = df[df['source_type'] == 'customer'][['source', 'age', 'gender']].drop_duplicates().rename(
        columns={'source': 'customer_id'})
    gender_dummies = pd.get_dummies(customers['gender'], prefix='gender')
    customer_features = pd.concat([customers[['age']], gender_dummies], axis=1)

    # Products
    products = df[df['target_type'] == 'product'][['target', 'product_name']].drop_duplicates().rename(
        columns={'target': 'product_id'})
    product_features = pd.get_dummies(products['product_name'], prefix='product')

    # Regions
    regions = df[df['target_type'] == 'region'][['target', 'region_name']].drop_duplicates().rename(
        columns={'target': 'region_id'})
    region_features = pd.get_dummies(regions['region_name'], prefix='region')

    # --- 2. Create Integer Mappings for Node IDs ---

    customer_encoder = LabelEncoder()
    customers['customer_idx'] = customer_encoder.fit_transform(customers['customer_id'])

    product_encoder = LabelEncoder()
    products['product_idx'] = product_encoder.fit_transform(products['product_id'])

    region_encoder = LabelEncoder()
    regions['region_idx'] = region_encoder.fit_transform(regions['region_id'])

    # --- 3. Map Integer Indices back to the main DataFrame ---

    # Create a copy to avoid modifying the original DataFrame
    processed_df = df.copy()

    # Map customer indices
    cust_map = customers.set_index('customer_id')['customer_idx']
    processed_df['source_idx'] = processed_df['source'].map(cust_map)

    # Map product and region indices to the target column
    prod_map = products.set_index('product_id')['product_idx']
    reg_map = regions.set_index('region_id')['region_idx']

    # Apply maps based on target_type
    is_prod = processed_df['target_type'] == 'product'
    is_reg = processed_df['target_type'] == 'region'
    processed_df.loc[is_prod, 'target_idx'] = processed_df.loc[is_prod, 'target'].map(prod_map)
    processed_df.loc[is_reg, 'target_idx'] = processed_df.loc[is_reg, 'target'].map(reg_map)
    processed_df['target_idx'] = processed_df['target_idx'].astype(int)

    # --- 4. Return all processed components ---

    return {
        "customer_features": customer_features,
        "product_features": product_features,
        "region_features": region_features,
        "interactions": processed_df,
        "encoders": {
            "customer": customer_encoder,
            "product": product_encoder,
            "region": region_encoder
        }
    }


def to_pyg_data(processed_data):
    data = HeteroData()

    # 1. Add node features
    data['customer'].x = torch.tensor(processed_data['customer_features'].values, dtype=torch.float)
    data['product'].x = torch.tensor(processed_data['product_features'].values, dtype=torch.float)
    data['region'].x = torch.tensor(processed_data['region_features'].values, dtype=torch.float)

    # 2. Add edges and edge attributes by filtering the interactions DataFrame
    interactions_df = processed_data['interactions']

    # Customer -> lives_in -> Region edges
    lives_in_df = interactions_df[interactions_df['edge_type'] == 'lives_in']
    data['customer', 'lives_in', 'region'].edge_index = torch.tensor([
        lives_in_df['source_idx'].values,
        lives_in_df['target_idx'].values
    ], dtype=torch.long)

    # Customer -> buys -> Product edges
    buys_df = interactions_df[interactions_df['edge_type'] == 'buys']
    data['customer', 'buys', 'product'].edge_index = torch.tensor([
        buys_df['source_idx'].values,
        buys_df['target_idx'].values
    ], dtype=torch.long)
    data['customer', 'buys', 'product'].edge_attr = torch.tensor(
        buys_df[['quantity', 'price']].values, dtype=torch.float
    )

    # Customer -> viewed -> Product edges
    viewed_df = interactions_df[interactions_df['edge_type'] == 'viewed']
    data['customer', 'viewed', 'product'].edge_index = torch.tensor([
        viewed_df['source_idx'].values,
        viewed_df['target_idx'].values
    ], dtype=torch.long)
    data['customer', 'viewed', 'product'].edge_attr = torch.tensor(
        viewed_df[['duration_seconds']].values, dtype=torch.float
    )

    print("\n--- PyTorch Geometric HeteroData Object Created ---")

    return data


def visualize_graph(G, path='c:/workspace/gnn/customer_graph2.html'):
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
    net.save_graph(path)




