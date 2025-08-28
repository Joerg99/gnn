import torch

def get_device(device_str):
    return torch.device(device_str if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def evaluate(model, data, mask, device):
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=1)
    acc = (pred[mask] == data.y[mask]).sum().float() / mask.sum()
    return acc.item()