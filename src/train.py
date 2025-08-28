import torch
from src.data.loaders import load_dataset
from src.models.gnn_models import GCNModel
from src.utils.helpers import get_device, evaluate
from torch_geometric.datasets import Planetoid

from pathlib import Path
import shutil







def train(config):
    device = get_device(config['device'])
    # Load data

    folder = Path(config['dataset']['root'])
    if folder.exists() and folder.is_dir():
        print(f"Deleting folder: {folder}")
        shutil.rmtree(folder)
    data = Planetoid(name=config['dataset']['name'], root=config['dataset']['root'])

    data = data.to(device)  # Move to GPU if available

    # Initialize model
    model = GCNModel(
        in_channels=data.num_features,
        hidden_channels=config['model']['hidden_channels'],
        out_channels=data.y.unique().shape[0],
        dropout=config['training']['dropout']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(config['training']['epochs']):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            train_acc = evaluate(model, data, data.train_mask, device)
            val_acc = evaluate(model, data, data.val_mask, device)
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    # Final test
    test_acc = evaluate(model, data, data.test_mask, device)
    print(f'Test Accuracy: {test_acc:.4f}')

    # Save model (e.g., torch.save(model.state_dict(), 'saved_models/model.pth'))
    return model

if __name__ == '__main__':
    import yaml

    with open('../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config

    print()