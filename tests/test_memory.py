from dataclasses import fields

import torch

from robotrehab.memory import load_memory, save_memory, subsample


def test_split_sizes(memory):
    assert len(memory.obs) == 400
    assert len(memory.val_obs) == 100


def test_memory_has_no_reward_field():
    from robotrehab.memory import Memory

    names = {f.name for f in fields(Memory)}
    assert names == {"obs", "actions", "val_obs", "val_actions", "unit_mean", "unit_std"}
    assert not any("reward" in name or "return" in name for name in names)


def test_statistics_come_from_the_healthy_network(healthy, memory):
    obs = torch.cat([memory.obs, memory.val_obs])
    with torch.no_grad():
        h1, h2 = healthy.hidden(obs)
    assert torch.allclose(memory.unit_std["h1"], h1.std(0), atol=1e-5)
    assert torch.allclose(memory.unit_mean["h2"], h2.mean(0), atol=1e-5)


def test_subsample_keeps_proportion_and_statistics(memory):
    small = subsample(memory, n_total=50, n_repair=40, seed=0)
    assert len(small.obs) == 40
    assert len(small.val_obs) == 10
    assert small.unit_std is memory.unit_std


def test_save_and_load(memory, tmp_path):
    save_memory(memory, tmp_path / "m.pt")
    loaded = load_memory(tmp_path / "m.pt")
    assert torch.equal(loaded.obs, memory.obs)
    assert torch.equal(loaded.unit_std["h1"], memory.unit_std["h1"])
