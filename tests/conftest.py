import pytest
import torch

from robotrehab.memory import build_memory
from robotrehab.policy import MaskedPolicy


@pytest.fixture
def healthy():
    torch.manual_seed(0)
    return MaskedPolicy(obs_dim=8, act_dim=3, hidden=32).eval()


@pytest.fixture
def memory(healthy):
    obs = torch.randn(500, 8, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        actions = healthy(obs)
    return build_memory(healthy, obs, actions, n_repair=400, seed=0)
