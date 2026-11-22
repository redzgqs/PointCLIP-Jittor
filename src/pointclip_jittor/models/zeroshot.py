"""PointCLIP V1: fixed six-view projection and CLIP image–text classification."""
import jittor as jt
from jittor import nn
import numpy as np

from ..encoders.text import TextEncoder, tokenize_modelnet40
from ..encoders.visual import ModifiedResNet
from ..encoders.weights import load_tower
from ..geometry.views import project_views


def normalize(x):
    return x / (x * x).sum(dim=-1, keepdims=True).sqrt()


PROTOCOLS = {
    "repo": (128, (1, 1, 1, 1, 1, 1)),
    "table1": (121, (3, 9, 5, 4, 5, 4)),
}


class ZeroShotPointCLIP:
    def __init__(self, weight_file, classes, upstream, protocol="repo"):
        self.resolution, self.relative_weights = PROTOCOLS[protocol]
        self.view_weights = jt.array(self.relative_weights).float32().reshape(1, 6, 1)
        self.visual = ModifiedResNet()
        text = TextEncoder()
        with np.load(weight_file, allow_pickle=False) as weights:
            load_tower(self.visual, weights, "visual.")
            load_tower(text, weights, "")
            self.scale = float(np.exp(weights["logit_scale"].astype(np.float32)).item())
        self.tokens = tokenize_modelnet40(classes, upstream)
        self.text_raw = text(self.tokens).sync()
        # The same 40 prompts apply to every object. Encode just once per run.
        self.class_vectors = normalize(self.text_raw.repeat(1, 6)).sync()

    def __call__(self, points):
        depths = project_views(points, resolution=self.resolution)
        maps = nn.interpolate(depths.unsqueeze(1), size=(224, 224), mode="bilinear",
                              align_corners=True)
        images = maps.repeat(1, 3, 1, 1)
        visual = self.visual(images)
        # Normalize each view before concatenating, exactly as in the upstream code.
        features = normalize(visual).reshape(points.shape[0], 6, 1024)
        weighted = (features * self.view_weights).reshape(points.shape[0], 6 * 1024)
        scores = (self.scale * weighted) @ self.class_vectors.transpose(1, 0)
        return scores, features
