"""Load CLIP image and text encoder weights from the converted NumPy archive."""
import numpy as np


def load_tower(module, weights, prefix):
    """Match every destination tensor by name and shape before running inference."""
    state = module.state_dict()
    values = {}
    for name, var in state.items():
        key = prefix + name
        if key not in weights:
            raise KeyError(f"Missing CLIP weight: {key}")
        value = weights[key]
        if tuple(var.shape) != value.shape:
            raise ValueError(f"{key}: expected {var.shape}, found {value.shape}")
        values[name] = value.astype(np.float32, copy=False)
    module.load_parameters(values)
    module.eval()
