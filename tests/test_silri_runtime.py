import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.silri.configuration_silri import SiLRIConfig
from lerobot.policies.silri.modeling_silri import MultivariateNormalDiag, SiLRIPolicy


def make_policy(images=False, num_discrete_actions=None):
    features = {'observation.state': PolicyFeature(FeatureType.STATE, (7,))}
    if images:
        features.update({f'observation.images.{key}': PolicyFeature(FeatureType.VISUAL, (3, 128, 128))
                         for key in ('left_wrist', 'right_aux')})
    cfg = SiLRIConfig(input_features=features,
                      output_features={'action': PolicyFeature(FeatureType.ACTION, (6,))},
                      device='cpu', use_torch_compile=False, shared_encoder=False,
                      normalization_mapping={}, dataset_stats={}, freeze_vision_encoder=False,
                      latent_dim=32, num_discrete_actions=num_discrete_actions)
    return SiLRIPolicy(cfg)


def batch(images=False):
    obs = {'observation.state': torch.randn(2, 7)}
    if images:
        obs.update({f'observation.images.{key}': torch.rand(2, 3, 128, 128)
                    for key in ('left_wrist', 'right_aux')})
    return dict(state=obs, next_state=obs, action=torch.zeros(2, 6),
                reward=torch.zeros(2), done=torch.zeros(2), is_intervention=torch.ones(2))


def test_diagonal_distribution_scale_is_standard_deviation():
    scale = torch.tensor([[.05, .1, .2]])
    dist = MultivariateNormalDiag(loc=torch.zeros_like(scale), scale_diag=scale)

    torch.testing.assert_close(dist.stddev, scale)
    torch.testing.assert_close(dist.covariance_matrix, torch.diag_embed(scale.square()))


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


def test_empty_intervention_mask_has_zero_finite_discrete_actor_loss():
    policy = make_policy(num_discrete_actions=2)
    actions = torch.cat((torch.zeros(2, 6), torch.tensor([[0.0], [1.0]])), dim=1)
    loss = policy.compute_loss_discrete_actor(
        observations=batch()['state'],
        old_actions=actions,
        is_intervention=torch.zeros(2),
    )['loss_actor']

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in policy.discrete_actor.parameters())


def test_explicit_bc_weight_sets_the_imitation_share():
    """bc_weight adds to the Lagrange multiplier inside the same normalisation."""
    torch.set_num_threads(2)
    policy = make_policy()
    data = batch()

    plain = policy(dict(data), model='actor')
    assert plain['explicit_bc_weight'] == 0.0
    # w = 0 must reproduce the original formulation exactly.
    weighted = policy(dict(data, bc_weight=0.0), model='actor')
    torch.testing.assert_close(weighted['loss_actor'], plain['loss_actor'])

    # The share is the per-sample mean of (lambda + w)/(1 + lambda + w), so it
    # must grow with w and stay below the pure-imitation limit of 1.
    previous = plain['imitation_share']
    for weight in (0.5, 1.5, 4.0):
        out = policy(dict(data, bc_weight=weight), model='actor')
        assert out['explicit_bc_weight'] == weight
        assert 0.0 <= previous < out['imitation_share'] < 1.0
        previous = out['imitation_share']
    huge = policy(dict(data, bc_weight=1e6), model='actor')
    assert huge['imitation_share'] > 0.999
    with pytest.raises(ValueError):
        policy(dict(data, bc_weight=-1.0), model='actor')


def test_bc_weight_moves_the_loss_towards_pure_imitation():
    """A large weight makes the actor loss approach the BC deviation."""
    torch.set_num_threads(2)
    policy = make_policy()
    data = batch()
    bc_loss = policy(dict(data), model='actor')['bc_loss']
    weight = 10_000.0
    heavy = policy(dict(data, bc_weight=weight), model='actor')
    # loss = (min_q + w*bc)/(1 + w) -> bc with an O(|min_q|/w) residual.
    bound = abs(heavy['min_q_preds']) / weight + 1e-4
    assert abs(heavy['loss_actor'].item() - bc_loss) < max(bound, 5e-4)
