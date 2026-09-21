import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.silri.configuration_silri import SiLRIConfig
from lerobot.policies.silri.modeling_silri import SiLRIPolicy


def make_policy(images=False):
    features = {'observation.state': PolicyFeature(FeatureType.STATE, (7,))}
    if images:
        features.update({f'observation.images.{key}': PolicyFeature(FeatureType.VISUAL, (3, 128, 128))
                         for key in ('left_wrist', 'right_aux')})
    cfg = SiLRIConfig(input_features=features,
                      output_features={'action': PolicyFeature(FeatureType.ACTION, (6,))},
                      device='cpu', use_torch_compile=False, shared_encoder=False,
                      normalization_mapping={}, dataset_stats={}, freeze_vision_encoder=False,
                      latent_dim=32)
    return SiLRIPolicy(cfg)


def batch(images=False):
    obs = {'observation.state': torch.randn(2, 7)}
    if images:
        obs.update({f'observation.images.{key}': torch.rand(2, 3, 128, 128)
                    for key in ('left_wrist', 'right_aux')})
    return dict(state=obs, next_state=obs, action=torch.zeros(2, 6),
                reward=torch.zeros(2), done=torch.zeros(2), is_intervention=torch.ones(2))


def test_fixed_gripper_optimizers_and_all_losses():
    torch.set_num_threads(2)
    policy = make_policy()
    optimizers, _ = policy.get_optimizer_and_scheduler()
    assert set(optimizers) == {'actor', 'critic', 'expert', 'lagrange'}
    for name in ('expert', 'actor_bc', 'critic', 'actor', 'lagrange'):
        optimizer = optimizers['actor' if name == 'actor_bc' else name]
        optimizer.zero_grad()
        loss = policy(batch(), model=name)['loss_' + name]
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for group in optimizer.param_groups
                   for p in group['params'] if p.grad is not None)
        optimizer.step()


def test_empty_human_mask_finite():
    policy = make_policy()
    data = batch()
    data['is_intervention'].zero_()
    for name in ('expert', 'actor_bc'):
        assert policy(data, model=name)['loss_' + name].item() == 0


def test_target_parameters_are_independent():
    policy = make_policy()
    assert not ({id(p) for p in policy.actor.parameters()} &
                {id(p) for p in policy.actor_target.parameters()})
    assert not ({id(p) for p in policy.critic_ensemble.parameters()} &
                {id(p) for p in policy.critic_target.parameters()})


def test_dual_rgb_update():
    torch.set_num_threads(2)
    policy = make_policy(images=True)
    optimizers, _ = policy.get_optimizer_and_scheduler()
    for name in ('expert', 'critic', 'actor', 'lagrange'):
        optimizers[name].zero_grad()
        loss = policy(batch(images=True), model=name)['loss_' + name]
        assert torch.isfinite(loss)
        loss.backward()
        optimizers[name].step()
