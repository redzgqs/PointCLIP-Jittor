"""Export OpenAI's official CLIP RN50 TorchScript checkpoint to named NumPy arrays."""

import argparse
import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from scripts import ROOT
RN50_SHA256 = 'afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762'


def convert(source, output):
    import torch

    digest = hashlib.sha256()
    with source.open('rb') as archive:
        for chunk in iter(lambda: archive.read(4 * 1024 ** 2), b''):
            digest.update(chunk)
    if digest.hexdigest() != RN50_SHA256:
        raise ValueError('The source differs from the official OpenAI CLIP RN50 archive: ' + digest.hexdigest())
    if source.resolve() == output.resolve():
        raise ValueError('Input and output must be distinct files')

    state = torch.jit.load(str(source), map_location='cpu').state_dict()
    # Includes learnable tensors, batch norm running statistics and JIT scalar buffers.
    arrays = {name: value.detach().cpu().numpy() for name, value in state.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=str(output.parent), suffix='.npz', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez(temporary, **arrays)
        with np.load(temporary, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays):
                raise ValueError('Checkpoint keys changed during conversion')
            for name, expected in arrays.items():
                actual = saved[name]
                if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
                    raise ValueError('Checkpoint array changed during conversion: ' + name)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print('Exported {} arrays to {} (source sha256 {})'.format(len(arrays), output, RN50_SHA256))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='PointCLIP project root')
    parser.add_argument('--input', type=Path, default=Path('weights/clip/RN50.pt'))
    parser.add_argument('--output', type=Path, default=Path('weights/clip/RN50.npz'))
    args = parser.parse_args()
    convert(args.root / args.input, args.root / args.output)


if __name__ == '__main__':
    main()
