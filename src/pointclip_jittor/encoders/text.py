"""ModelNet40 prompt tokenization and the Jittor text tower of OpenAI CLIP."""

import importlib.util
from pathlib import Path

import jittor as jt
from jittor import nn
import numpy as np


CONTEXT_LENGTH = 77
VOCAB_SIZE = 49408
WIDTH = 512
HEADS = 8
LAYERS = 12
EMBED_DIM = 1024
MODELNET40_TEMPLATE = "point cloud depth map of a {}."
FEWSHOT_TEMPLATE = "point cloud of a big {}."


def tokenize_modelnet40(classnames, pointclip_root, template=MODELNET40_TEMPLATE):
    """Use PointCLIP's original BPE vocabulary, keeping the given class order."""
    source = Path(pointclip_root) / "clip" / "simple_tokenizer.py"
    spec = importlib.util.spec_from_file_location("pointclip_bpe", source)
    module = importlib.util.module_from_spec(spec)
    # Load only the standalone tokenizer; the package initializer imports PyTorch.
    spec.loader.exec_module(module)
    tokenizer = module.SimpleTokenizer(str(source.with_name("bpe_simple_vocab_16e6.txt.gz")))
    start = tokenizer.encoder["<|startoftext|>"]
    end = tokenizer.encoder["<|endoftext|>"]
    rows = np.zeros((len(classnames), CONTEXT_LENGTH), dtype=np.int32)
    for index, name in enumerate(classnames):
        prompt = template.format(name.replace("_", " "))
        ids = [start, *tokenizer.encode(prompt), end]
        if len(ids) > CONTEXT_LENGTH:
            raise ValueError(f"CLIP prompt exceeds {CONTEXT_LENGTH} tokens: {prompt!r}")
        rows[index, :len(ids)] = ids
    return jt.array(rows)


class CausalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # The checkpoint packs all three affine maps along the output axis.
        self.in_proj_weight = jt.zeros((3 * WIDTH, WIDTH))
        self.in_proj_bias = jt.zeros((3 * WIDTH,))
        self.out_proj = nn.Linear(WIDTH, WIDTH)

    def execute(self, x, mask):
        batch, length, width = x.shape
        head_dim = width // HEADS
        qkv = nn.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.reshape((batch, length, 3, HEADS, head_dim)).permute(2, 0, 3, 1, 4)
        query, key, value = (qkv[i].reshape((batch * HEADS, length, head_dim)) for i in range(3))
        scores = nn.bmm(query * (head_dim ** -0.5), key.permute(0, 2, 1))
        weights = nn.softmax(scores + mask, dim=-1)
        output = nn.bmm(weights, value).reshape((batch, HEADS, length, head_dim))
        return self.out_proj(output.permute(0, 2, 1, 3).reshape((batch, length, width)))


class TextMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(WIDTH, 4 * WIDTH)
        self.c_proj = nn.Linear(4 * WIDTH, WIDTH)

    def execute(self, x):
        x = self.c_fc(x)
        return self.c_proj(x * jt.sigmoid(1.702 * x))  # CLIP QuickGELU


class ResidualAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = CausalAttention()
        self.ln_1 = nn.LayerNorm(WIDTH)
        self.ln_2 = nn.LayerNorm(WIDTH)
        self.mlp = TextMLP()

    def execute(self, x, mask):
        x = x + self.attn(self.ln_1(x), mask)
        return x + self.mlp(self.ln_2(x))


class TextTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.resblocks = nn.Sequential(*(ResidualAttentionBlock() for _ in range(LAYERS)))

    def execute(self, x):
        length = x.shape[1]
        mask = jt.array(np.triu(np.full((length, length), -np.inf, dtype=np.float32), k=1))
        for block in self.resblocks:
            x = block(x, mask)
        return x


class TextEncoder(nn.Module):
    """Load CLIP text weights with their original names; return raw EOT features."""

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, WIDTH)
        self.positional_embedding = jt.zeros((CONTEXT_LENGTH, WIDTH))
        self.transformer = TextTransformer()
        self.ln_final = nn.LayerNorm(WIDTH)
        self.text_projection = jt.zeros((WIDTH, embed_dim))

    def execute(self, token_ids):
        if token_ids.ndim != 2 or token_ids.shape[1] != CONTEXT_LENGTH:
            raise ValueError(f"Expected [classes, {CONTEXT_LENGTH}] tokens, got {token_ids.shape}")
        x = self.token_embedding(token_ids) + self.positional_embedding
        x = self.ln_final(self.transformer(x))
        # OpenAI's end-of-text token (49407) is the largest ID in each row.
        eot_position, _ = jt.argmax(token_ids, dim=1)
        eot = x[jt.arange(token_ids.shape[0]), eot_position]
        return jt.matmul(eot, self.text_projection)
