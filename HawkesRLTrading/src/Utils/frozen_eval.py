"""Load an explicit checkpoint and detect any weight changes during evaluation."""
import hashlib
import json
from pathlib import Path

import torch


def weight_digest(agent):
    digest = hashlib.sha256()
    for name in ('d', 'u'):
        for key, value in sorted(getattr(agent, 'Actor_Critic_' + name).state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            if not torch.isfinite(tensor).all():
                raise ValueError('Non-finite checkpoint weights')
            digest.update((name + key + str(tensor.dtype) + str(tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_frozen_checkpoint(agent, metadata_path):
    if not metadata_path:
        raise ValueError('Frozen policy evaluation requires EVAL_CHECKPOINT')
    path = Path(metadata_path).resolve()
    metadata = json.loads(path.read_text())
    files = {}
    for name in ('d', 'u'):
        model = getattr(agent, 'Actor_Critic_' + name)
        weights = Path(metadata['models'][name])
        if not weights.is_absolute():
            weights = path.parent / weights
        state = torch.load(weights, map_location=next(model.parameters()).device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        files[name] = hashlib.sha256(weights.read_bytes()).hexdigest()
    return {'metadata': str(path), 'epoch': metadata['epoch'],
            'file_sha256': files, 'weight_sha256': weight_digest(agent)}


def assert_frozen(agent, info):
    if weight_digest(agent) != info['weight_sha256']:
        raise RuntimeError('Checkpoint weights changed during frozen evaluation')
