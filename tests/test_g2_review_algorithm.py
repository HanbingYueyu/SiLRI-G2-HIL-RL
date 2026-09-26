from types import SimpleNamespace as NS
import pytest
import torch
from lerobot.policies.silri.modeling_silri import SiLRIPolicy


def test_squared_distance_and_point_one_slack():
    model = torch.tensor([[.3, .4, 0, 0, 0, 0]])
    zero = torch.zeros_like(model)
    class Expert:
        def get_dist(self, *args):
            return None, zero, zero
        def __call__(self, *args):
            return None, None, zero
    fake = NS(expert_network=Expert(), actor=lambda *args: (None, None, model),
              lagrange_net=lambda *args, **kwargs: torch.ones(1,1),
              critic_forward=lambda **kwargs: torch.zeros(2,1))
    loss, distance, _, violation = SiLRIPolicy.compute_loss_lagrange(fake, {})
    assert distance == pytest.approx(.25)
    assert violation == pytest.approx(.15)
    assert loss.item() == pytest.approx(-.15)
    assert SiLRIPolicy.compute_loss_actor(fake, {})['loss_actor'].item() == pytest.approx(.125)
