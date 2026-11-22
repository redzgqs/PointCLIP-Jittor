"""PointCLIP V1 inter-view adapter and matched RN101 classification heads."""
import jittor as jt
from jittor import nn
import numpy as np


VIEWS = 10
CHANNELS = 512


class BatchNormFeatures(nn.BatchNorm1d):
    """Use biased variance to normalize a batch, and unbiased variance in the running state."""

    def execute(self, x):
        if not self.is_train:
            return super().execute(x)
        n = x.shape[0]
        if n < 2:
            raise ValueError("Training batch normalization requires at least two samples")
        mean = x.mean(0)
        centered = x - mean
        variance = (centered * centered).mean(0)
        self.running_mean.update(self.running_mean * (1 - self.momentum) + self.momentum * mean.detach())
        unbiased = variance.detach() * (n / (n - 1))
        self.running_var.update(self.running_var * (1 - self.momentum) + self.momentum * unbiased)
        return centered * (self.weight / (variance + self.eps).sqrt()) + self.bias


class ViewBatchNorm(nn.Module):
    def __init__(self):
        self.bn = BatchNormFeatures(CHANNELS)

    def execute(self, x):
        return self.bn(x.reshape(-1, CHANNELS)).reshape(x.shape)


class InterViewAdapter(nn.Module):
    """Map ten raw RN101 features into one residual 5,120-dimensional descriptor."""

    def __init__(self):
        self.fusion_ratio = jt.full((VIEWS,), 0.5)
        self.global_f = nn.Sequential(
            ViewBatchNorm(), nn.Dropout(0.075), nn.Flatten(),
            nn.Linear(VIEWS * CHANNELS, CHANNELS), BatchNormFeatures(CHANNELS),
            nn.ReLU(), nn.Dropout(0.075),
        )
        self.view_f = nn.Sequential(
            nn.Linear(CHANNELS, CHANNELS), nn.ReLU(),
            nn.Linear(CHANNELS, VIEWS * CHANNELS), nn.ReLU(),
        )

    def execute(self, views):
        if views.ndim != 3 or tuple(views.shape[1:]) != (VIEWS, CHANNELS):
            raise ValueError("Expected raw RN101 view features [batch, 10, 512]")
        identity = views.reshape(views.shape[0], VIEWS * CHANNELS)
        global_feature = self.global_f(views * self.fusion_ratio.reshape(1, VIEWS, 1))
        return self.view_f(global_feature) * 0.6 + identity * 0.4


def unit_rows(x):
    return x / (x * x).sum(dim=-1, keepdims=True).sqrt()


def class_weights(text):
    """Apply the few-shot text template upstream, then share it across the ten views."""
    if tuple(text.shape) != (40, CHANNELS):
        raise ValueError("Expected 40 unnormalized RN101 text features of width 512")
    return unit_rows(text.repeat(1, VIEWS))


def score_views(features, text, logit_scale, adapter=None):
    """Use the same normalization and scoring regardless of which adapter is selected."""
    weights = class_weights(text) if text.shape[1] == CHANNELS else text
    if tuple(weights.shape) != (40, VIEWS * CHANNELS):
        raise ValueError("Expected text [40,512] or normalized repeated embeddings [40,5120]")
    if features.ndim != 3 or tuple(features.shape[1:]) != (VIEWS, CHANNELS):
        raise ValueError("Expected raw RN101 view features [batch,10,512]")
    image = features.reshape(features.shape[0], VIEWS * CHANNELS) if adapter is None else adapter(features)
    return (unit_rows(image) * logit_scale) @ weights.transpose(1, 0)


def load_adapter_npz(model, path):
    """Load weights exported from the authors' checkpoint or saved by this trainer."""
    state = model.state_dict()
    counters = {"global_f.0.bn.num_batches_tracked", "global_f.4.num_batches_tracked"}
    with np.load(path, allow_pickle=False) as saved:
        missing = set(state) - set(saved.files)
        unexpected = set(saved.files) - set(state) - counters
        if missing or unexpected:
            raise ValueError(f"Adapter state mismatch: missing={missing}, unexpected={unexpected}")
        params = {}
        for name, value in state.items():
            array = saved[name]
            if tuple(value.shape) != array.shape:
                raise ValueError(f"Shape mismatch for {name}: {array.shape} vs {tuple(value.shape)}")
            params[name] = array.astype(np.float32, copy=False)
    model.load_parameters(params)
    return model
