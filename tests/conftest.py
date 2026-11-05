import pytest
import torch

from robotrehab.policy import MaskedPolicy


@pytest.fixture
def healthy():
    torch.manual_seed(0)
    return MaskedPolicy(obs_dim=8, act_dim=3, hidden=32).eval()
