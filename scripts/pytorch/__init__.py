"""PyTorch controls based on the original PointCLIP implementation."""

RN50_VIEWS = 6
RN50_CHANNELS = 1024


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()
