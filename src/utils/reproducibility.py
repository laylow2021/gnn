import torch
import numpy as np
import random
import os
import joblib

def set_seed(seed: int = 42):
    """Rigorous seeding for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_device(config):
    """Determine the device (CPU/GPU) based on config and availability."""
    target = config.get('model', {}).get('device', 'auto').lower()
    if target == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if target == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device('cpu')

def save_checkpoint(model: torch.nn.Module, config: Dict[str, Any], path_prefix: str, 
                    node_scaler=None, edge_scaler=None):
    """Save model, scalers, and config for future inference."""
    os.makedirs(path_prefix, exist_ok=True)
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, os.path.join(path_prefix, 'model.pt'))

    if node_scaler:
        joblib.dump(node_scaler, os.path.join(path_prefix, 'node_scaler.joblib'))
    if edge_scaler:
        joblib.dump(edge_scaler, os.path.join(path_prefix, 'edge_scaler.joblib'))

    import yaml
    with open(os.path.join(path_prefix, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    print(f"Checkpoint saved to {path_prefix}")

def load_checkpoint(model_class, path_prefix: str, in_channels, edge_in_channels, device=None):
    """Load model, scalers, and config from checkpoint."""
    import yaml
    with open(os.path.join(path_prefix, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    if device is None:
        device = get_device(config)

    model = model_class(in_channels, edge_in_channels, config)
    model.load_state_dict(torch.load(os.path.join(path_prefix, 'model.pt'), map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    node_scaler = None
    if os.path.exists(os.path.join(path_prefix, 'node_scaler.joblib')):
        node_scaler = joblib.load(os.path.join(path_prefix, 'node_scaler.joblib'))

    edge_scaler = None
    if os.path.exists(os.path.join(path_prefix, 'edge_scaler.joblib')):
        edge_scaler = joblib.load(os.path.join(path_prefix, 'edge_scaler.joblib'))

    return model, config, node_scaler, edge_scaler