"""Jittor implementation of OpenAI's CLIP RN50 and RN101 visual encoders.

Reference architecture: https://github.com/openai/CLIP/blob/main/clip/model.py
"""

from collections import OrderedDict

import jittor as jt
from jittor import nn


class _VisualConv(nn.Conv2d):
    """Use patch matrices and cuBLAS for inference on CUDA installations without cuDNN."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)

    def execute(self, x):
        if (self.is_training() or not jt.flags.use_cuda or getattr(jt.compile_extern, "cudnn_ops", None)
                or not getattr(jt.compile_extern, "cublas_ops", None)):
            return super().execute(x)

        batch, channels, height, width = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        out_h = (height + 2 * ph - kh) // sh + 1
        out_w = (width + 2 * pw - kw) // sw + 1
        if self.kernel_size == (1, 1) and self.stride == (1, 1) and self.padding == (0, 0):
            patches = x.transpose(0, 2, 3, 1)
        else:
            # reindex supplies zeros outside the image, matching convolution padding.
            patches = x.reindex(
                [batch, out_h, out_w, channels, kh, kw],
                ["i0", "i3", f"i1*{sh}+i4-{ph}", f"i2*{sw}+i5-{pw}"],
            )
        rows = patches.reshape((batch * out_h * out_w, channels * kh * kw))
        filters = self.weight.reshape((self.out_channels, channels * kh * kw))
        result = jt.compile_extern.cublas_ops.cublas_matmul(rows, filters, False, True)
        return result.reshape((batch, out_h, out_w, self.out_channels)).transpose(0, 3, 1, 2)


class Bottleneck(jt.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = _VisualConv(in_channels, channels, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = _VisualConv(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = _VisualConv(channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride > 1 or in_channels != out_channels:
            # CLIP names the projection and its batch norm "0" and "1".
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride) if stride > 1 else nn.Identity()),
                ("0", _VisualConv(in_channels, out_channels, 1)),
                ("1", nn.BatchNorm2d(out_channels)),
            ]))

    def execute(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        y = nn.relu(self.bn1(self.conv1(x)))
        y = nn.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(self.avgpool(y)))
        return nn.relu(y + residual)


class AttentionPool2d(jt.Module):
    def __init__(self, spatial_size, channels, heads, output_dim):
        super().__init__()
        self.heads = heads
        self.head_dim = channels // heads
        self.positional_embedding = jt.randn(spatial_size ** 2 + 1, channels) * channels ** -0.5
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.c_proj = nn.Linear(channels, output_dim)

    def execute(self, feature_map):
        batch, channels, height, width = feature_map.shape
        patches = feature_map.reshape((batch, channels, height * width)).transpose(0, 2, 1)
        tokens = jt.concat([patches.mean(1, True), patches], dim=1)
        tokens = tokens + self.positional_embedding.unsqueeze(0)

        # Only the global token supplies a query. All 50 tokens supply keys and values.
        token_count = tokens.shape[1]
        query = self.q_proj(tokens[:, :1, :]).reshape((batch, 1, self.heads, self.head_dim))
        keys = self.k_proj(tokens).reshape((batch, token_count, self.heads, self.head_dim))
        values = self.v_proj(tokens).reshape((batch, token_count, self.heads, self.head_dim))
        scores = ((query * self.head_dim ** -0.5) * keys).sum(-1)
        weights = nn.softmax(scores, dim=1)
        pooled = (weights.unsqueeze(-1) * values).sum(1).reshape((batch, channels))
        return self.c_proj(pooled)


def _stage(in_channels, channels, blocks, stride):
    layers = [Bottleneck(in_channels, channels, stride)]
    for _ in range(1, blocks):
        layers.append(Bottleneck(channels * Bottleneck.expansion, channels))
    return nn.Sequential(*layers)


class ModifiedResNet(jt.Module):
    """Encode 224px RGB inputs into the shared image-text embedding space."""

    input_resolution = 224
    output_dim = 1024

    def __init__(self, backbone="RN50"):
        super().__init__()
        if backbone not in {"RN50", "RN101"}:
            raise ValueError(f"Unsupported CLIP ResNet backbone: {backbone}")
        self.output_dim = 512 if backbone == "RN101" else 1024
        self.conv1 = _VisualConv(3, 32, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = _VisualConv(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = _VisualConv(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.avgpool = nn.AvgPool2d(2)
        self.layer1 = _stage(64, 64, 3, stride=1)
        self.layer2 = _stage(256, 128, 4, stride=2)
        self.layer3 = _stage(512, 256, 23 if backbone == "RN101" else 6, stride=2)
        self.layer4 = _stage(1024, 512, 3, stride=2)
        self.attnpool = AttentionPool2d(7, 2048, 32, self.output_dim)

    def execute(self, images):
        if tuple(images.shape[1:]) != (3, self.input_resolution, self.input_resolution):
            raise ValueError(f"Expected [B, 3, 224, 224] images, got {images.shape}")
        x = nn.relu(self.bn1(self.conv1(images)))
        x = nn.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(nn.relu(self.bn3(self.conv3(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.attnpool(x)
