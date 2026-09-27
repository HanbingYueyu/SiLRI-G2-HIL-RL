from types import SimpleNamespace

import torch

from lerobot.policies.silri.modeling_silri import SiLRIPolicy


def test_checkpoint_copy_does_not_block_parameter_heartbeat(tmp_path, monkeypatch):
    import threading
    import g2_local.real_learner as module
    from test_g2_real_learner import _learner
    learner = _learner()
    original = module.deepcopy
    completed = threading.Event()
    threads = []
    def copying(value):
        if isinstance(value, dict) and 'online_replay' in value:
            def heartbeat():
                learner.heartbeat_parameters()
                completed.set()
            thread = threading.Thread(target=heartbeat)
            threads.append(thread)
            thread.start()
            assert completed.wait(1), 'Replay copy blocked the parameter heartbeat'
        return original(value)
    monkeypatch.setattr(module, 'deepcopy', copying)
    try:
        learner.save_checkpoint(tmp_path / 'checkpoint.pt')
    finally:
        for thread in threads:
            thread.join(2)


def test_target_critic_uses_executable_action_bounds():
    seen = []
    def critic_forward(**kwargs):
        if kwargs['use_target']:
            seen.append(kwargs['actions'])
        return torch.zeros(2, 1)
    policy = SimpleNamespace(
        actor_target=lambda *args: (torch.tensor([[2., -2., .3, 0., 0., 0.]]),),
        critic_forward=critic_forward, continuous_action_dim=6,
        config=SimpleNamespace(num_subsample_critics=None, discount=.99),
    )
    SiLRIPolicy.compute_loss_critic(policy, {}, torch.zeros(1, 6),
                                 torch.zeros(1), {}, torch.zeros(1))
    assert torch.allclose(seen[0], torch.tensor([[1-1e-6, -1+1e-6, .3, 0., 0., 0.]]))
