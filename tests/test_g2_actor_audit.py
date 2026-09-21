"""Real-inference audit orchestration with offline reader, clock and probe."""
import builtins
from dataclasses import replace
import importlib
import json
from pathlib import Path

import pytest

from g2_local.actor_inference import InferenceResult
from test_g2_freshness_audit import Rig, audit


class ActorRig(Rig):
    def __init__(self):
        super().__init__()
        self.events = []
        self.warmup_steps = 0
        self.warmup_completed = 0
        self.observations = []
        self.result_fault = lambda result: result
        self.probe_failure = None

    def observe(self):
        self.events.append('observe')
        super().observe()
        obs = {'observation_id': self.reads}
        self.observations.append(obs)
        return obs

    def client(self):
        client = super().client()
        original_read = client.read
        def read():
            self.events.append('snapshot')
            return original_read()
        client.read = read
        return client

    def probe(self, checkpoint, *, device, warmup_steps):
        self.created.append('probe')
        assert checkpoint == Path('trusted.pt') and device == 'cuda'
        self.warmup_steps = warmup_steps
        rig = self
        class Probe:
            def metadata(self):
                return dict(checkpoint_schema=1, checkpoint_version=3,
                            checkpoint_sha256='a'*64, policy_config={'device': 'cuda'},
                            gpu_name='NVIDIA GeForce RTX 3090', device='cuda',
                            warmup_steps=rig.warmup_steps,
                            warmup_completed=rig.warmup_completed)

            def warmup(self, obs):
                assert obs is rig.observations[0]
                for _ in range(rig.warmup_steps):
                    rig.events.append('warmup')
                    rig.now += 1_000_000_000
                    rig.warmup_completed += 1

            def infer(self, obs):
                assert obs is rig.observations[-1]
                assert obs is not rig.observations[0]
                rig.events.append('infer_start')
                if rig.probe_failure is not None:
                    raise rig.probe_failure
                rig.now += 20_000_000
                rig.events.append('infer_end')
                return rig.result_fault(InferenceResult(
                    cpu_prepare_ns=2_000_000, h2d_resize_ns=3_000_000,
                    forward_ns=10_000_000, total_ns=20_000_000,
                    cuda_allocated_bytes=2*1024**2, cuda_reserved_bytes=4*1024**2,
                    cuda_peak_allocated_bytes=3*1024**2, action_shape=(1, 6),
                    action_min=-.5, action_max=.25))

            def close(self):
                rig.closed.append('probe')
        return Probe()

    def run_actor(self, audit, output, **kwargs):
        return self.run(audit, output, actor_checkpoint=Path('trusted.pt'),
                        device='cuda', probe_factory=self.probe, **kwargs)


def test_formal_boundaries_and_unique_audit_identity_are_raw_evidence(audit, tmp_path):
    reports = []
    for index in range(2):
        output = tmp_path/str(index)
        report = ActorRig().run_actor(audit, output, warmup_steps=2)
        rows = [json.loads(line) for line in (output/'evidence.jsonl').read_text().splitlines()]
        start = next(row for row in rows if row['event'] == 'formal_start')
        end = next(row for row in rows if row['event'] == 'formal_end')
        assert rows[0]['audit_session_id'] == report['audit_session_id']
        assert report['formal_start_mono_ns'] == start['formal_start_mono_ns']
        assert start['initial_snapshot']['sequence'] < next(r for r in rows if r['event'] == 'sample')['snapshot']['sequence']
        assert report['formal_end_mono_ns'] == end['formal_end_mono_ns']
        assert report['formal_elapsed_s'] == 30.
        assert end['formal_elapsed_s'] == 30.
        assert start['formal_start_mono_ns'] >= next(r for r in rows if r['event'] == 'warmup')['end_mono_ns']
        reports.append(report)
    assert reports[0]['audit_session_id'] != reports[1]['audit_session_id']


def test_snapshot_and_age_are_measured_after_real_inference(audit, tmp_path, monkeypatch):
    rig = ActorRig()
    original = audit._measure
    def measure(*args):
        rig.events.append('measure')
        return original(*args)
    monkeypatch.setattr(audit, '_measure', measure)
    report = rig.run_actor(audit, tmp_path/'audit', warmup_steps=2)
    assert rig.events[:4] == ['snapshot', 'observe', 'warmup', 'warmup']
    formal = rig.events[4:]
    assert len(formal) == report['sample_count']*5
    assert formal == ['observe', 'infer_start', 'infer_end', 'snapshot', 'measure']*report['sample_count']
    assert report['camera_age_ms']['left_wrist']['min'] == 32.
    assert report['actions_discarded'] is True
    assert report['accepted_count'] == report['sample_count']
    assert report['rejected_count'] == 0
    assert set(rig.closed) == {'reader', 'client', 'probe'}


def test_warmup_is_separate_and_scalar_evidence_has_literal_units(audit, tmp_path):
    rig = ActorRig()
    output = tmp_path/'audit'
    report = rig.run_actor(audit, output, warmup_steps=2)
    assert report['warmup_steps'] == report['warmup_completed'] == 2
    assert rig.reads == report['sample_count']+1
    assert report['sample_count'] == 400  # 30 seconds, 5ms read + 20ms infer + 50ms pause.
    for metric, value in [('cpu_prepare_ms', 2.), ('h2d_resize_ms', 3.),
                          ('actor_forward_ms', 10.), ('actor_inference_ms', 20.),
                          ('cuda_allocated_mib', 2.), ('cuda_reserved_mib', 4.),
                          ('cuda_peak_allocated_mib', 3.)]:
        assert report[metric] == dict(min=value, p50=value, p95=value, p99=value, max=value)
    assert report['action_min'] == -.5 and report['action_max'] == .25
    assert report['actor_metadata']['checkpoint_sha256'] == 'a'*64
    records = [json.loads(line) for line in (output/'evidence.jsonl').read_text().splitlines()]
    warmup = [row for row in records if row['event'] == 'warmup']
    assert len(warmup) == 1 and warmup[0]['warmup_completed'] == 2
    samples = [row for row in records if row['event'] == 'sample']
    assert samples[0]['inference']['action_shape'] == [1, 6]
    assert samples[0]['inference']['action_discarded'] is True
    assert samples[0]['inference']['cuda_allocated_bytes'] == 2*1024**2
    assert all(len(line.encode())+1 <= audit._MAX_ROW_BYTES
               for line in (output/'evidence.jsonl').read_text().splitlines())
    assert json.loads((output/'summary.json').read_text()) == report


@pytest.mark.parametrize('kwargs', [
    {'device': None}, {'device': 'cpu'}, {'warmup_steps': -1},
    {'warmup_steps': True}, {'warmup_steps': 1.5}, {'inference_delay_s': .01},
])
def test_invalid_actor_options_fail_before_any_resource(audit, tmp_path, kwargs):
    rig = ActorRig()
    options = dict(actor_checkpoint=Path('trusted.pt'), device='cuda',
                   probe_factory=rig.probe)
    options.update(kwargs)
    with pytest.raises(ValueError):
        rig.run(audit, tmp_path/'audit', **options)
    assert rig.created == []
    assert not (tmp_path/'audit').exists()


@pytest.mark.parametrize('kwargs', [{'device': 'cuda'}, {'warmup_steps': 1},
                                  {'probe_factory': lambda: None}])
def test_actor_only_options_require_checkpoint(audit, tmp_path, kwargs):
    rig = ActorRig()
    with pytest.raises(ValueError):
        rig.run(audit, tmp_path/'audit', **kwargs)
    assert rig.created == []


def test_existing_output_rejected_before_probe_construction(audit, tmp_path):
    rig = ActorRig()
    with pytest.raises(FileExistsError):
        rig.run_actor(audit, tmp_path)
    assert rig.created == []


@pytest.mark.parametrize('failure', [ValueError('bad action'), SystemExit(0), KeyboardInterrupt()])
def test_formal_failure_counts_rejection_and_closes_resources(audit, tmp_path, failure):
    rig = ActorRig()
    rig.probe_failure = failure
    if isinstance(failure, KeyboardInterrupt):
        result = rig.run_actor(audit, tmp_path/'audit')
        assert result['status'] == 'interrupted'
    else:
        with pytest.raises((ValueError, RuntimeError)):
            rig.run_actor(audit, tmp_path/'audit')
    result = json.loads((tmp_path/'audit'/'summary.json').read_text())
    assert result['sample_count'] == result['accepted_count'] == 0
    assert result['rejected_count'] == 1
    assert result['actions_discarded'] is True
    assert set(rig.closed) == {'reader', 'client', 'probe'}


@pytest.mark.parametrize('fault', [
    lambda r: replace(r, action_discarded=False),
    lambda r: replace(r, total_ns=float('nan')),
    lambda r: replace(r, action_min=float('inf')),
])
def test_invalid_probe_diagnostics_never_publish_nonfinite_json(audit, tmp_path, fault):
    rig = ActorRig()
    rig.result_fault = fault
    with pytest.raises(ValueError):
        rig.run_actor(audit, tmp_path/'audit')
    report = json.loads((tmp_path/'audit'/'summary.json').read_text())
    assert report['status'] == 'failed' and report['rejected_count'] == 1
    assert 'NaN' not in (tmp_path/'audit'/'evidence.jsonl').read_text()
    assert 'Infinity' not in (tmp_path/'audit'/'evidence.jsonl').read_text()


def test_actor_mode_never_imports_motion_gym_or_learner(audit, tmp_path, monkeypatch):
    original = builtins.__import__
    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        assert not set(fromlist or ()) & {'GdkCommandPort', 'MotionBackend',
                                          'G2LocalEnv', 'LearnerServiceStub'}
        assert not any(part in name for part in ('motion_backend', 'gymnasium', 'learner_client'))
        return original(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, '__import__', guarded)
    importlib.reload(audit)
    importlib.reload(importlib.import_module('g2_local.actor_inference'))
    assert ActorRig().run_actor(audit, tmp_path/'audit')['motion_authorized'] is False


def test_cli_routes_actor_options_after_validation(audit, tmp_path, monkeypatch):
    rig = ActorRig()
    monkeypatch.setattr(audit, '_actor_probe', rig.probe, raising=False)
    monkeypatch.setattr(audit, 'SnapshotClient', lambda *args, **kwargs: rig.client())
    monkeypatch.setattr(audit, '_reader', rig.reader)
    monkeypatch.setattr(audit.time, 'monotonic_ns', lambda: rig.now)
    monkeypatch.setattr(audit.time, 'sleep', rig.sleep)
    args = ['--socket', 'offline.sock', '--seconds', '30', '--actor-checkpoint',
            'trusted.pt', '--device', 'cuda', '--warmup-steps', '2']
    assert audit.main(args+['--output', str(tmp_path)]) == 1
    assert rig.created == []
    assert audit.main(args+['--output', str(tmp_path/'audit')]) == 0
    assert json.loads((tmp_path/'audit'/'summary.json').read_text())['warmup_completed'] == 2


@pytest.mark.parametrize('failure', [ValueError('warmup failed'), SystemExit(0), KeyboardInterrupt()])
def test_partial_warmup_counts_survive_failure_without_formal_samples(audit, tmp_path, failure):
    rig = ActorRig()
    original = rig.probe
    def factory(*args, **kwargs):
        probe = original(*args, **kwargs)
        def warmup(obs):
            rig.warmup_completed = 1
            raise failure
        probe.warmup = warmup
        return probe
    rig.probe = factory
    if isinstance(failure, KeyboardInterrupt):
        rig.run_actor(audit, tmp_path/'audit', warmup_steps=2)
    else:
        with pytest.raises((ValueError, RuntimeError)):
            rig.run_actor(audit, tmp_path/'audit', warmup_steps=2)
    report = json.loads((tmp_path/'audit'/'summary.json').read_text())
    assert report['warmup_completed'] == 1
    assert report['warmup_steps'] == 2
    assert report['sample_count'] == report['accepted_count'] == report['rejected_count'] == 0
    records = [json.loads(line) for line in (tmp_path/'audit'/'evidence.jsonl').read_text().splitlines()]
    warmup = [row for row in records if row['event'] == 'warmup']
    assert len(warmup) == 1 and warmup[0]['warmup_completed'] == 1
    assert set(rig.closed) == {'reader', 'client', 'probe'}


@pytest.mark.parametrize('stage', ['construct', 'cleanup'])
@pytest.mark.parametrize('failure', [ValueError('probe failed'), SystemExit(0), KeyboardInterrupt()])
def test_probe_baseexceptions_preserve_session_and_cleanup(audit, tmp_path, stage, failure):
    rig = ActorRig()
    original = rig.probe
    def factory(*args, **kwargs):
        if stage == 'construct':
            raise failure
        probe = original(*args, **kwargs)
        def close():
            rig.closed.append('probe')
            raise failure
        probe.close = close
        return probe
    rig.probe = factory
    if stage == 'construct' and isinstance(failure, KeyboardInterrupt):
        report = rig.run_actor(audit, tmp_path/'audit')
        assert report['status'] == 'interrupted'
    else:
        with pytest.raises((ValueError, RuntimeError)):
            rig.run_actor(audit, tmp_path/'audit')
    report = json.loads((tmp_path/'audit'/'summary.json').read_text())
    assert report['status'] != 'completed'
    assert report['actions_discarded'] is True
    assert report['accepted_count'] == report['sample_count']
    assert report['rejected_count'] == 0
    assert rig.closed[-1] == 'client'


def test_nonfinite_final_metadata_cannot_corrupt_summary(audit, tmp_path):
    rig = ActorRig()
    original = rig.probe
    def factory(*args, **kwargs):
        probe = original(*args, **kwargs)
        metadata = probe.metadata
        def metadata_after_warmup():
            result = metadata()
            if rig.warmup_completed:
                result['bad_metric'] = float('nan')
            return result
        probe.metadata = metadata_after_warmup
        return probe
    rig.probe = factory
    with pytest.raises((ValueError, RuntimeError)):
        rig.run_actor(audit, tmp_path/'audit', warmup_steps=2)
    report = json.loads((tmp_path/'audit'/'summary.json').read_text())
    assert report['status'] == 'failed'
    assert 'NaN' not in (tmp_path/'audit'/'summary.json').read_text()
    assert set(rig.closed) == {'reader', 'client', 'probe'}


def test_warmup_evidence_limit_failure_keeps_exact_completed_count(audit, tmp_path, monkeypatch):
    rig = ActorRig()
    original = rig.probe
    def factory(*args, **kwargs):
        probe = original(*args, **kwargs)
        warmup = probe.warmup
        def finish(obs):
            warmup(obs)
            monkeypatch.setattr(audit, '_MAX_LOG_BYTES', 1)
        probe.warmup = finish
        return probe
    rig.probe = factory
    with pytest.raises(ValueError, match='evidence limit'):
        rig.run_actor(audit, tmp_path/'audit', warmup_steps=2)
    report = json.loads((tmp_path/'audit'/'summary.json').read_text())
    assert report['status'] == 'failed' and report['warmup_completed'] == 2
    assert report['accepted_count'] == report['rejected_count'] == 0


def test_action_extrema_aggregate_all_accepted_samples(audit, tmp_path):
    rig = ActorRig()
    def varying(result):
        if rig.reads == 2:
            return replace(result, action_min=-.7, action_max=.8)
        return result
    rig.result_fault = varying
    report = rig.run_actor(audit, tmp_path/'audit')
    assert report['action_min'] == -.7 and report['action_max'] == .8
