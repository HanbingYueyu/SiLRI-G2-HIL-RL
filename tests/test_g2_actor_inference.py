"""Probe contract tests: serialization is real; only CUDA and network work are faked."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
from hashlib import sha256
from types import SimpleNamespace
import gc
import weakref

import draccus
import numpy as np
import pytest
import torch

from g2_local import actor_inference
from g2_local.policy import create_policy


@pytest.fixture(scope='module')
def config():
    torch.set_num_threads(2)
    policy = create_policy('cpu')
    encoded = draccus.encode(policy.config)
    encoded['device'] = 'cuda'
    return encoded


class FakePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(7, 6)
        self.output = torch.zeros(1, 6)
        self.calls = 0

    def select_action(self, batch):
        assert torch.is_inference_mode_enabled()
        self.batch = batch
        self.calls += 1
        return self.output, {}


@pytest.fixture
def rig(tmp_path, monkeypatch, config):
    cuda = SimpleNamespace(name='NVIDIA GeForce RTX 3090', available=True,
                           synchronizations=0, releases=0)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: cuda.available)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda device: cuda.name)

    def synchronize(device=None):
        cuda.synchronizations += 1

    def empty_cache():
        cuda.releases += 1

    monkeypatch.setattr(torch.cuda, 'synchronize', synchronize)
    monkeypatch.setattr(torch.cuda, 'empty_cache', empty_cache)
    for name, value in [('memory_allocated', 128), ('memory_reserved', 256),
                        ('max_memory_allocated', 192)]:
        monkeypatch.setattr(torch.cuda, name, lambda device=None, value=value: value)
    policy = FakePolicy()

    def factory(device):
        assert device == 'cuda'
        return policy

    monkeypatch.setattr(actor_inference, 'create_policy', factory)
    snapshot = {'schema': 1, 'version': 3, 'config': deepcopy(config),
                'policy': {'actor.weight': torch.ones(6, 7),
                           'actor.bias': torch.zeros(6),
                           'critic.unused': torch.full((1,), float('nan'))}}
    path = tmp_path / 'trusted-fixture.pt'

    def build(*, device='cuda', warmup_steps=2):
        torch.save(snapshot, path)
        return actor_inference.ActorInferenceProbe(path, device=device,
                                                  warmup_steps=warmup_steps)

    return SimpleNamespace(build=build, cuda=cuda, policy=policy,
                           snapshot=snapshot, path=path)


@pytest.mark.parametrize('fault', ['schema', 'schema_bool', 'version', 'version_bool',
                                  'camera', 'state', 'action', 'config_bool', 'type',
                                  'gpu', 'cuda_unavailable', 'missing_actor_keys',
                                  'extra_actor_keys', 'actor_shape', 'actor_dtype',
                                  'actor_nonfinite', 'actor_nontensor'])
def test_rejects_invalid_checkpoint_config_gpu_and_actor_weights(rig, fault):
    cfg = rig.snapshot['config']
    if fault.startswith('schema'):
        rig.snapshot['schema'] = True if fault.endswith('bool') else 2
    elif fault.startswith('version'):
        rig.snapshot['version'] = True if fault.endswith('bool') else -1
    elif fault == 'camera':
        del cfg['input_features']['observation.images.right_aux']
    elif fault == 'state':
        cfg['input_features']['observation.state']['shape'] = [6]
    elif fault == 'action':
        cfg['output_features']['action']['shape'] = [7]
    elif fault == 'config_bool':
        cfg['shared_encoder'] = 0
    elif fault == 'type':
        cfg['type'] = 'sac'
    elif fault == 'gpu':
        rig.cuda.name = 'NVIDIA GeForce RTX 4090'
    elif fault == 'cuda_unavailable':
        rig.cuda.available = False
    elif fault == 'missing_actor_keys':
        del rig.snapshot['policy']['actor.bias']
    elif fault == 'extra_actor_keys':
        rig.snapshot['policy']['actor.extra'] = torch.zeros(1)
    elif fault == 'actor_shape':
        rig.snapshot['policy']['actor.bias'] = torch.zeros(7)
    elif fault == 'actor_dtype':
        rig.snapshot['policy']['actor.bias'] = torch.zeros(6, dtype=torch.float64)
    elif fault == 'actor_nonfinite':
        rig.snapshot['policy']['actor.bias'][0] = float('inf')
    else:
        rig.snapshot['policy']['actor.bias'] = [0] * 6
    with pytest.raises(ValueError):
        rig.build()


@pytest.mark.parametrize('device', ['cpu', 'cuda:0', 'mps', None, True])
def test_rejects_devices_other_than_explicit_cuda(rig, device):
    with pytest.raises(ValueError, match='device'):
        rig.build(device=device)


@pytest.mark.parametrize('steps', [-1, True, 1.0, '2'])
def test_requires_nonnegative_integer_warmup_count(rig, steps):
    with pytest.raises(ValueError, match='warmup'):
        rig.build(warmup_steps=steps)


def test_checkpoint_hash_version_and_config_are_defensive_metadata(rig):
    probe = rig.build()
    meta = probe.metadata()
    assert meta['checkpoint_sha256'] == sha256(rig.path.read_bytes()).hexdigest()
    assert meta['checkpoint_schema'] == 1
    assert meta['checkpoint_version'] == 3
    assert meta['gpu_name'] == 'NVIDIA GeForce RTX 3090'
    assert meta['policy_config'] == rig.snapshot['config']
    assert not rig.policy.training
    assert torch.equal(rig.policy.actor.weight, torch.ones(6, 7))
    meta['policy_config']['input_features'].clear()
    rig.path.write_bytes(b'replaced after construction')
    assert probe.metadata()['policy_config']['input_features']
    assert probe.metadata()['checkpoint_sha256'] == meta['checkpoint_sha256']
    probe.close()
    probe.close()


def test_trusted_checkpoint_loaded_on_cpu_with_explicit_pickle_boundary(rig, monkeypatch):
    original_load = torch.load
    calls = []

    def load(path, **kwargs):
        calls.append(kwargs)
        return original_load(path, **kwargs)

    monkeypatch.setattr(torch, 'load', load)
    rig.build().close()
    assert calls == [{'map_location': 'cpu', 'weights_only': False}]


@pytest.fixture
def observation():
    image = np.empty((1056, 1280, 3), dtype=np.uint8)
    image[:] = [0, 127, 255]
    return {'state': np.arange(7, dtype=np.float32),
            'left_wrist': image, 'right_aux': image.copy()}


@pytest.fixture
def inference(rig, monkeypatch):
    original_to = torch.Tensor.to
    transfers = []

    def transfer(tensor, *args, **kwargs):
        if args and args[0] == 'cuda':
            transfers.append((tuple(tensor.shape), tensor.dtype))
            args = ('cpu',) + args[1:]
        elif kwargs.get('device') == 'cuda':
            transfers.append((tuple(tensor.shape), tensor.dtype))
            kwargs['device'] = 'cpu'
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'to', transfer)
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda tensor: True))
    probe = rig.build()
    yield SimpleNamespace(probe=probe, rig=rig, transfers=transfers)
    probe.close()


def test_full_frame_pipeline_times_synchronized_work_and_discards_action(inference, observation):
    probe = inference.probe
    result = probe.infer(observation)
    batch = inference.rig.policy.batch
    assert batch['observation.state'].shape == (1, 7)
    assert torch.equal(batch['observation.state'], torch.arange(7).reshape(1, 7))
    for key in ('left_wrist', 'right_aux'):
        image = batch[f'observation.images.{key}']
        assert image.shape == (1, 3, 128, 128)
        assert image.dtype == torch.float32
        torch.testing.assert_close(image[0, :, 32, 32], torch.tensor([0., 127 / 255, 1.]))
    assert inference.transfers.count(((1, 3, 1056, 1280), torch.float32)) == 2
    assert inference.rig.cuda.synchronizations >= 3
    assert result.action_shape == (1, 6)
    assert result.action_discarded is True
    assert result.action_min == result.action_max == 0.0
    assert result.total_ns >= result.cpu_prepare_ns + result.h2d_resize_ns + result.forward_ns
    assert result.forward_ns > 0
    for name, value in asdict(result).items():
        if name.endswith(('_ns', '_bytes')):
            assert type(value) is int and value >= 0
    assert (result.cuda_allocated_bytes, result.cuda_reserved_bytes,
            result.cuda_peak_allocated_bytes) == (128, 256, 192)
    assert not any(isinstance(value, (torch.Tensor, np.ndarray)) for value in asdict(result).values())
    with pytest.raises(FrozenInstanceError):
        result.action_discarded = False


def test_resize_is_explicit_bilinear_with_no_corner_alignment(inference, observation, monkeypatch):
    original = torch.nn.functional.interpolate
    calls = []

    def resize(tensor, **kwargs):
        calls.append(kwargs)
        return original(tensor, **kwargs)

    monkeypatch.setattr(torch.nn.functional, 'interpolate', resize)
    inference.probe.infer(observation)
    assert calls == [dict(size=(128, 128), mode='bilinear', align_corners=False,
                          antialias=False)] * 2


@pytest.mark.parametrize('fault', ['missing', 'state_dtype', 'state_shape', 'state_nan',
                                  'state_noncontiguous', 'image_dtype', 'image_shape',
                                  'image_noncontiguous', 'image_list'])
def test_invalid_observation_closes_before_forward(inference, observation, fault):
    if fault == 'missing':
        del observation['right_aux']
    elif fault == 'state_dtype':
        observation['state'] = observation['state'].astype(np.float64)
    elif fault == 'state_shape':
        observation['state'] = np.zeros(6, dtype=np.float32)
    elif fault == 'state_nan':
        observation['state'][0] = np.nan
    elif fault == 'state_noncontiguous':
        observation['state'] = np.zeros(14, dtype=np.float32)[::2]
    elif fault == 'image_dtype':
        observation['left_wrist'] = observation['left_wrist'].astype(np.float32)
    elif fault == 'image_shape':
        observation['left_wrist'] = np.zeros((128, 128, 3), dtype=np.uint8)
    elif fault == 'image_noncontiguous':
        observation['left_wrist'] = observation['left_wrist'][:, ::-1]
    else:
        observation['left_wrist'] = []
    with pytest.raises(ValueError, match='observation'):
        inference.probe.infer(observation)
    assert inference.rig.policy.calls == 0
    assert inference.probe.policy is None
    assert inference.rig.cuda.releases == 1


@pytest.mark.parametrize('output', [torch.zeros(6), torch.zeros(1, 7),
                                   torch.full((1, 6), float('nan')),
                                   torch.full((1, 6), float('inf')),
                                   torch.ones(1, 6), -torch.ones(1, 6),
                                   torch.zeros(1, 6, dtype=torch.int32),
                                   torch.zeros(1, 6, dtype=torch.float64), [0] * 6])
def test_invalid_action_is_never_returned(inference, observation, output):
    inference.rig.policy.output = output
    with pytest.raises(ValueError, match='action'):
        inference.probe.infer(observation)
    assert inference.probe.policy is None


def test_clamp_boundary_is_accepted(inference, observation):
    inference.rig.policy.output = torch.tensor([[-1 + 1e-6, 1 - 1e-6, 0, 0, 0, 0]])
    result = inference.probe.infer(observation)
    assert -1 < result.action_min < 0 < result.action_max < 1


def test_cpu_action_is_rejected(inference, observation, monkeypatch):
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda tensor: False))
    with pytest.raises(ValueError, match='action'):
        inference.probe.infer(observation)


@pytest.mark.parametrize('error', [RuntimeError('CUDA fault'), KeyboardInterrupt(), SystemExit(7)])
@pytest.mark.parametrize('phase', ['transfer', 'forward', 'synchronize'])
def test_cuda_and_base_exceptions_close_probe(inference, observation, monkeypatch, error, phase):
    def fail(*args, **kwargs):
        raise error

    if phase == 'transfer':
        original_to = torch.Tensor.to

        def fail_cuda_transfer(tensor, *args, **kwargs):
            if (args and args[0] == 'cuda') or kwargs.get('device') == 'cuda':
                raise error
            return original_to(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, 'to', fail_cuda_transfer)
    elif phase == 'forward':
        monkeypatch.setattr(inference.rig.policy, 'select_action', fail)
    else:
        monkeypatch.setattr(torch.cuda, 'synchronize', fail)
    with pytest.raises(type(error)):
        inference.probe.infer(observation)
    assert inference.probe.policy is None
    assert inference.rig.cuda.releases == 1
    inference.probe.close()
    assert inference.rig.cuda.releases == 1


@pytest.mark.parametrize('error', [RuntimeError('partial construction'), KeyboardInterrupt()])
def test_partial_construction_releases_resources(rig, monkeypatch, error):
    def fail(device):
        raise error

    monkeypatch.setattr(actor_inference, 'create_policy', fail)
    with pytest.raises(type(error)):
        rig.build()
    assert rig.cuda.releases == 1


def test_warmup_counts_exact_steps_and_preserves_action_discard(inference, observation):
    assert inference.probe.warmup(observation) is None
    assert inference.rig.policy.calls == 2
    assert inference.probe.metadata()['warmup_completed'] == 2
    inference.probe.infer(observation)
    assert inference.rig.policy.calls == 3
    assert inference.probe.metadata()['warmup_completed'] == 2
    with pytest.raises(ValueError, match='warmup'):
        inference.probe.warmup(observation)


def test_warmup_baseexception_keeps_successful_count_and_closes(inference, observation, monkeypatch):
    original = inference.rig.policy.select_action

    def fail_second(batch):
        if inference.rig.policy.calls:
            raise KeyboardInterrupt()
        return original(batch)

    monkeypatch.setattr(inference.rig.policy, 'select_action', fail_second)
    with pytest.raises(KeyboardInterrupt):
        inference.probe.warmup(observation)
    assert inference.probe.metadata()['warmup_completed'] == 1
    assert inference.probe.policy is None


def test_closed_probe_rejects_inference(inference, observation):
    inference.probe.close()
    with pytest.raises(ValueError, match='closed'):
        inference.probe.infer(observation)


def test_cleanup_interrupt_is_reported_as_cleanup_failure(rig, monkeypatch):
    probe = rig.build()

    def fail():
        raise KeyboardInterrupt()

    monkeypatch.setattr(torch.cuda, 'empty_cache', fail)
    with pytest.raises(BaseException) as caught:
        probe.close()
    assert isinstance(caught.value, RuntimeError)
    assert 'cleanup failed' in str(caught.value)
    assert probe.policy is None
    probe.close()


def test_synchronization_delimits_reported_stages(inference, observation, monkeypatch):
    events = []
    times = iter([100, 110, 120, 150, 180, 200])
    original_to = torch.Tensor.to
    original_forward = inference.rig.policy.select_action

    def transfer(tensor, *args, **kwargs):
        if args and args[0] == 'cuda':
            events.append('transfer')
        return original_to(tensor, *args, **kwargs)

    def forward(batch):
        events.append('forward')
        return original_forward(batch)

    monkeypatch.setattr(actor_inference, 'monotonic_ns', lambda: next(times))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: events.append('sync'))
    monkeypatch.setattr(torch.Tensor, 'to', transfer)
    monkeypatch.setattr(inference.rig.policy, 'select_action', forward)
    result = inference.probe.infer(observation)
    assert events == ['sync', 'transfer', 'transfer', 'transfer', 'sync', 'forward', 'sync']
    assert (result.cpu_prepare_ns, result.h2d_resize_ns, result.forward_ns,
            result.total_ns) == (10, 30, 30, 100)


@pytest.mark.parametrize('value', [True, -1, 1.0])
def test_invalid_cuda_memory_statistics_fail_closed(inference, observation, monkeypatch, value):
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: value)
    with pytest.raises(ValueError, match='memory'):
        inference.probe.infer(observation)
    assert inference.probe.policy is None


def test_zero_warmup_performs_no_inference(rig):
    probe = rig.build(warmup_steps=0)
    assert probe.warmup({}) is None
    assert rig.policy.calls == 0
    assert probe.metadata()['warmup_completed'] == 0
    probe.close()


def test_actual_silri_actor_load_and_forward_on_cpu_with_simulated_cuda(
        inference, observation, monkeypatch):
    inference.probe.close()
    real_policy = create_policy('cpu')
    inference.rig.snapshot['policy'] = real_policy.state_dict()
    monkeypatch.setattr(actor_inference, 'create_policy', lambda device: real_policy)
    probe = inference.rig.build()
    try:
        result = probe.infer(observation)
        assert result.action_shape == (1, 6)
        assert result.action_discarded is True
        assert -1 < result.action_min <= result.action_max < 1
    finally:
        probe.close()


@pytest.mark.parametrize('phase', ['validation', 'factory'])
def test_constructor_failure_releases_tensors_with_exception_retained(rig, monkeypatch, phase):
    refs = []
    released_before_cache = []
    original_empty_cache = torch.cuda.empty_cache

    def empty_cache():
        released_before_cache.append(all(reference() is None for reference in refs))
        original_empty_cache()

    monkeypatch.setattr(torch.cuda, 'empty_cache', empty_cache)

    class TrackingLinear(torch.nn.Linear):
        def state_dict(self):
            state = super().state_dict()
            refs.extend(weakref.ref(tensor) for tensor in state.values())
            return state

    def factory(device):
        policy = FakePolicy()
        policy.actor = TrackingLinear(7, 6)
        refs.append(weakref.ref(policy))
        refs.extend(weakref.ref(parameter) for parameter in policy.parameters())
        if phase == 'factory':
            temporary = torch.zeros(8)
            refs.append(weakref.ref(temporary))
            raise KeyboardInterrupt('construction interrupted')
        return policy

    monkeypatch.setattr(actor_inference, 'create_policy', factory)
    rig.snapshot['policy']['actor.bias'][0] = float('nan')
    with pytest.raises(BaseException) as retained:
        rig.build()
    expected = ValueError if phase == 'validation' else KeyboardInterrupt
    assert type(retained.value) is expected
    assert retained.value.__traceback__ is not None
    # Keep the caller's exception and its traceback intact throughout the check.
    gc.collect()
    assert refs and all(reference() is None for reference in refs)
    assert released_before_cache == [True]
    assert rig.cuda.releases == 1


@pytest.mark.parametrize('phase', ['preprocessing', 'forward', 'action', 'chained_forward'])
def test_inference_failure_releases_tensors_with_exception_retained(
        inference, observation, monkeypatch, phase):
    refs = []
    original_resize = torch.nn.functional.interpolate
    original_cpu = torch.Tensor.cpu
    original_empty_cache = torch.cuda.empty_cache
    released_before_cache = []

    def cpu(tensor, *args, **kwargs):
        diagnostic = original_cpu(tensor, *args, **kwargs)
        refs.append(weakref.ref(diagnostic))
        return diagnostic

    def empty_cache():
        released_before_cache.append(all(reference() is None for reference in refs))
        original_empty_cache()

    def resize(tensor, **kwargs):
        refs.append(weakref.ref(tensor))
        if phase == 'preprocessing':
            temporary = tensor.clone()
            refs.append(weakref.ref(temporary))
            raise KeyboardInterrupt('resize interrupted')
        result = original_resize(tensor, **kwargs)
        refs.append(weakref.ref(result))
        return result

    def chained_failure(batch):
        temporary = batch['observation.state'].clone()
        refs.append(weakref.ref(temporary))
        raise RuntimeError('underlying CUDA error')

    def forward(batch):
        refs.extend(weakref.ref(tensor) for tensor in batch.values())
        action = torch.full((1, 6), float('nan'))
        refs.append(weakref.ref(action))
        if phase == 'forward':
            raise RuntimeError('forward failed')
        if phase == 'chained_forward':
            try:
                chained_failure(batch)
            except RuntimeError as error:
                raise ValueError('forward context') from error
        return action, {}

    monkeypatch.setattr(torch.nn.functional, 'interpolate', resize)
    monkeypatch.setattr(torch.Tensor, 'cpu', cpu)
    monkeypatch.setattr(torch.cuda, 'empty_cache', empty_cache)
    monkeypatch.setattr(inference.rig.policy, 'select_action', forward)
    with pytest.raises(BaseException) as retained:
        inference.probe.infer(observation)
    expected = {'preprocessing': KeyboardInterrupt, 'forward': RuntimeError,
                'action': ValueError, 'chained_forward': ValueError}[phase]
    assert type(retained.value) is expected
    assert retained.value.__traceback__ is not None
    if phase == 'chained_forward':
        assert type(retained.value.__cause__) is RuntimeError
        assert str(retained.value.__cause__) == 'underlying CUDA error'
    gc.collect()
    assert refs and all(reference() is None for reference in refs)
    assert released_before_cache == [True]
    assert inference.probe.policy is None
    assert inference.rig.cuda.releases == 1
