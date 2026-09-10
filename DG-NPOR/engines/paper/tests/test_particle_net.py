import importlib.util

import numpy as np
import pytest


torch_available = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not torch_available, reason="optional PyTorch dependency is absent")
def test_particle_net_forward_respects_padded_shape():
    import torch

    from jetset_litcomp.particle_net import ParticleNet

    model = ParticleNet(architecture="lite")
    tracks = torch.from_numpy(np.random.RandomState(1).normal(size=(3, 20, 19)).astype("f4"))
    mask = torch.zeros((3, 20), dtype=torch.bool)
    mask[:, :6] = True
    context = torch.zeros((3, 4), dtype=torch.float32)
    output = model(tracks, mask, context)
    assert output.shape == (3,)
    assert torch.isfinite(output).all()

