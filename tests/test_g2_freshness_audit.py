"""Offline audit contracts: no vendor SDK, PTP, or motion calls."""
import ast
from dataclasses import replace
import importlib
import json
from pathlib import Path
import stat

import pytest

from test_g2_freshness import NOW, ORIGIN, SOURCE_NOW, SOURCES, evidence, snapshot


def test_audit_module_available_for_read_only_load_collection():
    assert importlib.util.find_spec('g2_local.freshness_audit') is not None


@pytest.fixture
def audit():
    # Missing implementation is an assertion failure in the first RED run.
    assert importlib.util.find_spec('g2_local.freshness_audit') is not None
    return importlib.import_module('g2_local.freshness_audit')


class Rig:
    def __init__(self):
        self.now = NOW
        self.sequence = 0
        self.reads = 0
        self.closed = []
        self.created = []
        self.info_fault = lambda info: None
        self.snapshot_fault = lambda snap: snap
        self.interrupt = False
        self.last_info = {}

    def sleep(self, seconds):
        if self.interrupt:
            raise KeyboardInterrupt
        self.now += round(seconds * 1e9)

    def client(self):
        self.created.append('client')
        rig = self
        class Client:
            def read(self):
                rig.sequence += 1
                return rig.snapshot_fault(snapshot(
                    sequence=rig.sequence, reference_mono_ns=rig.now,
                    created_mono_ns=rig.now, last_sample_mono_ns=rig.now,
                    valid_until_ns=rig.now+2_500_000_000))

            def close(self):
                rig.closed.append('client')
        return Client()

    def reader(self):
        self.created.append('reader')
        return self

    def observe(self):
        self.reads += 1
        self.now += 5_000_000
        info = evidence(**dict.fromkeys(SOURCES, SOURCE_NOW+self.now-NOW-10_000_000))
        for name in ('read_start_monotonic_ns', 'read_end_monotonic_ns',
                     'read_start_wall_ns', 'read_end_wall_ns', 'read_start_sdk_clock_ns',
                     'read_end_sdk_clock_ns', 'sdk_clock_ns', 'received_monotonic_ns',
                     'state_received_monotonic_ns'):
            info[name] += self.now-NOW
        self.info_fault(info)
        self.last_info = info
        return {'unused_images': object()}

    def close(self):
        self.closed.append('reader')

    def run(self, audit, output, **kwargs):
        return audit.run_audit(output=output, duration_s=30,
                               client_factory=self.client, reader_factory=self.reader,
                               monotonic_ns=lambda: self.now, sleep=self.sleep, **kwargs)


@pytest.mark.parametrize('duration', [None, True, 29, 1801, 30., float('inf')])
def test_invalid_duration_rejected_before_factories(audit, tmp_path, duration):
    rig = Rig()
    with pytest.raises(ValueError):
        audit.run_audit(output=tmp_path/'new', duration_s=duration,
                        client_factory=rig.client, reader_factory=rig.reader)
    assert rig.created == []
    assert not (tmp_path/'new').exists()


@pytest.mark.parametrize('delay', [-.1, True, float('nan'), float('inf'), '1'])
def test_invalid_inference_delay_rejected_before_factories(audit, tmp_path, delay):
    rig = Rig()
    with pytest.raises(ValueError):
        rig.run(audit, tmp_path/'new', inference_delay_s=delay)
    assert rig.created == []


def test_existing_output_preserved_before_client_or_reader_creation(audit, tmp_path):
    marker = tmp_path/'original'
    marker.write_text('keep')
    rig = Rig()
    with pytest.raises(FileExistsError):
        rig.run(audit, tmp_path)
    assert rig.created == []
    assert marker.read_text() == 'keep'


def test_symlink_output_parent_cannot_redirect_evidence(audit, tmp_path):
    (tmp_path/'link').symlink_to(tmp_path, target_is_directory=True)
    rig = Rig()
    with pytest.raises(OSError):
        rig.run(audit, tmp_path/'link'/'new')
    assert rig.created == []
    assert not (tmp_path/'new').exists()


def test_summary_has_literal_units_and_intervals_without_threshold_approval(audit, tmp_path):
    rig = Rig()
    output = tmp_path/'audit'
    result = rig.run(audit, output, inference_delay_s=.02)
    assert result['status'] == 'completed'
    assert result['motion_authorized'] is False
    assert result['thresholds_approved'] is False
    assert result['camera_age_ms']['left_wrist'] == dict(min=32., p50=32., p95=32., p99=32., max=32.)
    assert result['camera_age_lower_ms']['left_wrist']['min'] == 28.
    assert result['state_age_ms']['joint']['p99'] == 32.
    assert result['camera_skew_ms']['max'] == 4.
    assert result['source_camera_skew_ms']['max'] == 0.
    assert result['gdk_read_duration_ms']['max'] == 5.
    assert result['inference_duration_ms']['min'] == 20.
    assert result['mapping_error_ms']['max'] == 2.
    assert set(result['mapping_drift_ppm']) == {'min', 'p50', 'p95', 'p99', 'max'}
    assert result['gdk_read_gap_ms']['min'] >= 20.
    assert result['snapshot_gap_ms']['min'] > 0.
    assert result['tf_position_error_m']['max'] == 0.
    assert result['tf_rotation_error_rad']['max'] == 0.
    assert rig.closed == ['reader', 'client']
    rows = [json.loads(line) for line in (output/'evidence.jsonl').read_text().splitlines()]
    samples = [row for row in rows if row['event'] == 'sample']
    assert len(samples) == result['sample_count']
    assert samples[0]['info']['source_timestamp_ns']['tf'] == SOURCE_NOW-5_000_000
    assert samples[0]['snapshot']['utc_offset_valid'] == 0
    assert samples[0]['source_intervals_ns']['tf'] == [NOW-7_000_000, NOW-3_000_000]
    assert json.loads((output/'summary.json').read_text()) == result
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE((output/'evidence.jsonl').stat().st_mode) == 0o600


def test_percentiles_interpolate_nonconstant_observed_distribution(audit):
    rows = [{'camera_age_ms': {'left_wrist': age}} for age in (0., 20., 40., 50.)]
    result = audit.summarize_audit(rows)
    assert result['camera_age_ms']['left_wrist'] == pytest.approx(dict(min=0., p50=30., p95=48.5, p99=49.7, max=50.))


def test_reused_mutable_reader_metadata_does_not_corrupt_previous_sample(audit, tmp_path):
    rig = Rig()
    original = rig.observe
    reused = {}
    def observe():
        obs = original()
        reused.clear()
        reused.update(rig.last_info)
        rig.last_info = reused
        return obs
    rig.observe = observe
    assert rig.run(audit, tmp_path/'audit')['status'] == 'completed'


def test_reader_close_failure_still_closes_client_and_marks_failed(audit, tmp_path):
    rig = Rig()
    def close():
        raise RuntimeError('reader cleanup failed')
    rig.close = close
    with pytest.raises(RuntimeError, match='cleanup_failed'):
        rig.run(audit, tmp_path/'audit')
    assert rig.closed == ['client']
    assert json.loads((tmp_path/'audit'/'summary.json').read_text())['status'] == 'failed'


def test_sample_count_bound_rejects_before_extra_reader_call(audit, tmp_path, monkeypatch):
    monkeypatch.setattr(audit, '_MAX_SAMPLES', 2)
    rig = Rig()
    with pytest.raises(ValueError, match='sample limit'):
        rig.run(audit, tmp_path/'audit')
    assert rig.reads == 2


def test_cli_requires_duration_and_existing_output_has_no_factories(audit, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('CLI touched read resources before validation')
    monkeypatch.setattr(audit, 'SnapshotClient', forbidden)
    monkeypatch.setattr(audit, '_reader', forbidden)
    with pytest.raises(SystemExit) as error:
        audit.main(['--socket', 'missing.sock'])
    assert error.value.code == 2
    assert audit.main(['--socket', 'missing.sock', '--seconds', '30', '--output', str(tmp_path)]) == 1


def test_cli_ctrl_c_returns_130_with_read_resources_closed(audit, tmp_path, monkeypatch):
    rig = Rig()
    rig.interrupt = True
    monkeypatch.setattr(audit, 'SnapshotClient', lambda *args, **kwargs: rig.client())
    monkeypatch.setattr(audit, '_reader', rig.reader)
    monkeypatch.setattr(audit.time, 'monotonic_ns', lambda: rig.now)
    monkeypatch.setattr(audit.time, 'sleep', rig.sleep)
    assert audit.main(['--socket', 'offline.sock', '--seconds', '30',
                       '--output', str(tmp_path/'audit')]) == 130
    assert rig.closed == ['reader', 'client']


@pytest.mark.parametrize('source', SOURCES)
@pytest.mark.parametrize('delta,reason', [(0, 'frozen'), (-1, 'reversed')])
def test_frozen_or_reversed_sources_fail_and_preserve_rejected_raw_evidence(audit, tmp_path, source, delta, reason):
    rig = Rig()
    first = {}
    def fault(info):
        if not first:
            first.update(info['source_timestamp_ns'])
        else:
            stamp = first[source]+delta
            info['source_timestamp_ns'][source] = stamp
            if source in SOURCES[:2]:
                info['camera_timestamp_ns'][source] = stamp
            if source == 'tf':
                info['tf_queries'][1]['timestamp_ns'] = stamp
    rig.info_fault = fault
    with pytest.raises(ValueError, match=reason):
        rig.run(audit, tmp_path/'audit')
    result = json.loads((tmp_path/'audit'/'summary.json').read_text())
    assert result['status'] == 'failed' and result['sample_count'] == 1
    rows = [json.loads(line) for line in (tmp_path/'audit'/'evidence.jsonl').read_text().splitlines()]
    assert rows[-1]['event'] == 'rejected'
    assert rows[-1]['info']['source_timestamp_ns'][source] == first[source]+delta
    assert rig.closed == ['reader', 'client']


@pytest.mark.parametrize('fault', [
    lambda s: replace(s, healthy=False),
    lambda s: replace(s, valid_until_ns=s.last_sample_mono_ns-1),
    lambda s: replace(s, sequence=1),
    lambda s: replace(s, offset_at_reference_ns=float('nan')),
    lambda s: replace(s, boot_id='wrong'),
])
def test_bad_snapshots_fail_closed(audit, tmp_path, fault):
    rig = Rig()
    rig.snapshot_fault = fault
    with pytest.raises(ValueError):
        rig.run(audit, tmp_path/'audit')
    assert json.loads((tmp_path/'audit'/'summary.json').read_text())['status'] == 'failed'
    assert rig.closed[-1] == 'client'


def test_disconnection_before_reader_creation(audit, tmp_path):
    rig = Rig()
    def fail(_):
        raise ConnectionError('monitor disconnected')
    rig.snapshot_fault = fail
    with pytest.raises(ConnectionError):
        rig.run(audit, tmp_path/'audit')
    assert rig.created == ['client']
    assert rig.closed == ['client']


@pytest.mark.parametrize('fault', [
    lambda i: i.pop('tf_queries'),
    lambda i: i['source_timestamp_ns'].update(joint=True),
    lambda i: i.update(tf_rotation_error_rad=float('nan')),
    lambda i: i.update(read_end_wall_ns=ORIGIN+NOW),
])
def test_invalid_complete_source_evidence_is_rejected(audit, tmp_path, fault):
    rig = Rig()
    rig.info_fault = fault
    with pytest.raises((ValueError, KeyError)):
        rig.run(audit, tmp_path/'audit')
    assert rig.closed == ['reader', 'client']


def test_ctrl_c_writes_interrupted_summary_and_closes_only_read_resources(audit, tmp_path):
    rig = Rig()
    rig.interrupt = True
    result = rig.run(audit, tmp_path/'audit', inference_delay_s=.1)
    assert result['status'] == 'interrupted'
    assert result['motion_authorized'] is False
    assert rig.closed == ['reader', 'client']


def test_evidence_byte_bound_fails_without_unbounded_file(audit, tmp_path, monkeypatch):
    monkeypatch.setattr(audit, '_MAX_LOG_BYTES', 10000)
    rig = Rig()
    with pytest.raises(ValueError, match='evidence limit'):
        rig.run(audit, tmp_path/'audit')
    assert (tmp_path/'audit'/'evidence.jsonl').stat().st_size <= 10000
    assert rig.closed == ['reader', 'client']


def test_audit_import_and_injected_run_never_load_motion_or_vendor_sdk(audit, tmp_path, monkeypatch):
    # Test the safety boundary both structurally and at the import boundary.
    source = Path(audit.__file__).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            assert all('GdkCommandPort' not in name.name and 'motion_backend' not in name.name
                       for name in node.names)
            assert 'motion_backend' not in (getattr(node, 'module', '') or '')
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name not in ('agibot_gdk', 'g2_local.motion_backend')
        assert 'GdkCommandPort' not in (kwargs.get('fromlist', ()) or ())
        if len(args) >= 3:
            assert 'GdkCommandPort' not in (args[2] or ())
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    importlib.reload(audit)
    assert Rig().run(audit, tmp_path/'audit')['motion_authorized'] is False
