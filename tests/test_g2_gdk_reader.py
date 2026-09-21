"""Offline reader contracts: SDK imports are replaced before construction."""
from copy import deepcopy
from types import ModuleType, SimpleNamespace as NS
import json
import sys

import numpy as np
import pytest

from g2_local.gdk_backend import GdkReader


class FakeTF:
    def __init__(self):
        self.calls = []
        self.failures = 0
        self.fail_target = None
        self.stamps = [1005, 1004]
        self.poses = [[-.1, -.2, -.3, 0., 0., 0., 1.],
                      [.1, .2, .3, 0., 0., 0., 1.]]

    def lookup_transform_latest(self, target, source, return_timestamp):
        assert return_timestamp is True
        assert (target, source) in [('base_link', 'arm_l_end_link'),
                                    ('arm_l_end_link', 'base_link')]
        self.calls.append((target, source))
        if self.failures or target == self.fail_target:
            self.failures = max(0, self.failures - 1)
            raise RuntimeError('Failed to lookup latest transform')
        index = int(target == 'arm_l_end_link')
        pose = self.poses[index]
        return NS(translation=NS(**dict(zip('xyz', pose[:3]))),
                  rotation=NS(**dict(zip('xyzw', pose[3:])))), self.stamps[index]


@pytest.fixture
def rig(monkeypatch, tmp_path):
    # Complete data at the boundary; no real adapter/GDK module is imported.
    monkeypatch.setattr(sys, 'path', list(sys.path))
    root = tmp_path / 'adapter'
    (root / 'g2_adapter').mkdir(parents=True)
    (root / 'g2_adapter' / 'control.py').touch()
    tf = FakeTF()
    state = NS(tf=tf, released=0, offset=0, fault=None,
               pose=NS(position_m=(.1, .2, .3), orientation_xyzw=(0., 0., 0., -1.)))

    def check(stage):
        if state.fault == stage:
            raise RuntimeError('failed ' + stage)

    def joint():
        check('joint')
        return {'timestamp': 1003 + state.offset, 'states': [
            {'name': f'arm_{side}_joint{i}', 'position': 0., 'velocity': 0.,
             'effort': 0.} for side in ('l', 'r') for i in range(1, 8)]}

    def frame(stream, timeout):
        check('camera_' + str(stream))
        return NS(timestamp_ns=1000 + stream + state.offset, width=3, height=2,
                  data=bytes([stream] * 18), encoding='rgb8')

    def decode(frame, gdk):
        check('decode')
        return np.frombuffer(frame.data, dtype=np.uint8).reshape(2, 3, 3)

    class Controller:
        def __init__(self, gdk, robot, *, allow_motion):
            assert allow_motion is False
        def checked_arm_state(self):
            check('arm')
            return (0.,) * 14
        def read_end_effector_pose(self, name):
            assert name == 'arm_l_end_link'
            check('pose')
            return state.pose
        def motion_status_summary(self):
            check('status')
            return {'control_mode': 1, 'error_code': 0}

    def sdk_clock():
        check('clock')
        return 1100 + state.offset

    def release():
        state.released += 1

    modules = {'g2_adapter': ModuleType('g2_adapter'),
               'g2_adapter.control': NS(G2Controller=Controller),
               'g2_adapter.camera': NS(decode_color_rgb=decode),
               'agibot_gdk': NS(gdk_init=lambda: 0, gdk_release=release,
                   GDKRes=NS(kSuccess=0), Robot=lambda: NS(get_joint_states=joint),
                   Camera=lambda streams: NS(get_latest_image=frame), TF=lambda: tf,
                   CameraType=NS(kHandLeftColor=1, kHandRightColor=2),
                   Clock=NS(now_ns=sdk_clock))}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    # A deterministic advancing local clock bounds retry tests without sleeping.
    now = [10.]
    def sleep(seconds):
        now[0] += seconds
    monkeypatch.setattr('g2_local.gdk_backend.time.sleep', sleep)
    monkeypatch.setattr('g2_local.gdk_backend.time.monotonic', lambda: now[0])
    readers = []
    def make():
        reader = GdkReader(adapter_root=root, timeout_s=.1)
        readers.append(reader)
        return reader
    state.make = make
    yield state
    for reader in readers:
        reader.close()


def test_observe_exposes_all_source_times_without_changing_policy_observation(rig):
    reader = rig.make()
    obs = reader.observe()
    assert set(obs) == {'state', 'left_wrist', 'right_aux'}
    assert obs['state'].shape == (7,) and obs['state'].dtype == np.float32
    np.testing.assert_allclose(obs['state'], [.1, .2, .3, 0, 0, 0, -1])
    for key, value in [('left_wrist', 1), ('right_aux', 2)]:
        assert obs[key].shape == (2, 3, 3) and obs[key].dtype == np.uint8
        assert (obs[key] == value).all()
    info = reader.last_info
    assert info['source_timestamp_ns'] == dict(left_wrist=1001, right_aux=1002,
                                               joint=1003, tf=1004)
    assert [q['timestamp_ns'] for q in info['tf_queries']] == [1005, 1004]
    assert info['tf_queries'][0]['pose'][:3] == [-.1, -.2, -.3]
    assert info['tf_queries'][1]['pose'][:3] == [.1, .2, .3]
    assert info['tf_position_error_m'] == 0
    assert info['tf_rotation_error_rad'] == 0
    assert info['read_start_monotonic_ns'] <= info['state_received_monotonic_ns']
    assert info['state_received_monotonic_ns'] <= info['read_end_monotonic_ns']
    assert info['read_start_wall_ns'] <= info['read_end_wall_ns']
    assert info['read_start_sdk_clock_ns'] == info['read_end_sdk_clock_ns'] == 1100
    assert info['sdk_clock_ns'] == 1100
    json.dumps(info, allow_nan=False)
    rig.offset = 100
    rig.tf.stamps = [1105, 1104]
    reader.observe()
    assert reader.last_info['source_timestamp_ns'] == dict(
        left_wrist=1101, right_aux=1102, joint=1103, tf=1104)
    assert info['source_timestamp_ns']['tf'] == 1004


@pytest.mark.parametrize('fault', ['arm', 'pose', 'camera_1', 'camera_2', 'decode',
                                  'joint', 'status', 'clock', 'tf_forward', 'tf_reverse'])
def test_failed_transaction_retains_only_previous_success_and_no_camera_commit(rig, fault):
    reader = rig.make()
    reader.observe()
    previous = reader.last_info
    snapshot = deepcopy(previous)
    stamps = dict(reader.last_stamps)
    rig.offset = 100
    rig.fault = fault
    if fault.startswith('tf_'):
        rig.tf.fail_target = 'base_link' if fault == 'tf_forward' else 'arm_l_end_link'
    with pytest.raises(RuntimeError):
        reader.observe()
    assert reader.last_info is previous and reader.last_info == snapshot
    assert reader.last_stamps == stamps


def test_reader_waits_for_both_tf_directions_on_startup(rig):
    rig.tf.failures = 2
    reader = rig.make()
    assert reader.tf is rig.tf
    assert rig.tf.calls == [('base_link', 'arm_l_end_link')] * 3 + [
        ('arm_l_end_link', 'base_link')]


@pytest.mark.parametrize('target', ['base_link', 'arm_l_end_link'])
def test_startup_timeout_releases_gdk_when_either_direction_missing(rig, target):
    rig.tf.fail_target = target
    with pytest.raises(TimeoutError, match='TF cache'):
        rig.make()
    assert rig.released == 1


@pytest.mark.parametrize('which', ['motion', 'tf_forward', 'tf_reverse'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), 0., 2.])
def test_invalid_quaternion_in_either_tf_or_motion_rejects_transaction(rig, which, value):
    reader = rig.make()
    if which == 'motion':
        rig.pose.orientation_xyzw = (0, 0, 0, value)
    else:
        rig.tf.poses[int(which == 'tf_reverse')][6] = value
    with pytest.raises(ValueError, match='pose'):
        reader.observe()
    assert reader.last_info == {} and reader.last_stamps == {}


def test_tf_errors_use_fixed_sdk_direction_even_if_other_direction_is_closer(rig):
    from g2_local.clock_probe import sample_gdk
    reader = rig.make()
    rig.tf.poses[0] = [.1, .2, .3, 0, 0, 0, -1]
    rig.tf.poses[1] = [.4, .6, .3, 0, 0, 1, 0]
    reader.observe()
    info = reader.last_info
    assert info['tf_position_error_m'] == pytest.approx(.5)
    assert info['tf_rotation_error_rad'] == pytest.approx(np.pi)
    row = sample_gdk(reader, reader.tf)
    assert row['tf_queries'] == info['tf_queries']
    assert row['timestamps']['tf'] == 1004
    assert row['tf_position_error_m'] == info['tf_position_error_m']
    assert row['tf_rotation_error_rad'] == info['tf_rotation_error_rad']


@pytest.mark.parametrize('source', ['left_wrist', 'right_aux', 'joint', 'tf_forward',
                                   'tf_reverse', 'clock'])
@pytest.mark.parametrize('bad', [True, 1000.5, '1000', 0, -1])
def test_invalid_source_timestamp_is_never_coerced_into_valid_evidence(rig, source, bad):
    reader = rig.make()
    if source in ('left_wrist', 'right_aux'):
        get_image = reader.camera.get_latest_image
        def invalid_frame(stream, timeout):
            frame = get_image(stream, timeout)
            if stream == (1 if source == 'left_wrist' else 2):
                frame.timestamp_ns = bad
            return frame
        reader.camera.get_latest_image = invalid_frame
    elif source == 'joint':
        reader.robot.get_joint_states = lambda: {'timestamp': bad, 'states': []}
    elif source == 'clock':
        reader.gdk.Clock.now_ns = lambda: bad
    else:
        rig.tf.stamps[int(source == 'tf_reverse')] = bad
    with pytest.raises(ValueError, match='timestamp'):
        reader.observe()
    assert reader.last_info == {} and reader.last_stamps == {}


def test_failure_at_final_sdk_clock_does_not_commit_partial_evidence(rig):
    reader = rig.make()
    reader.observe()
    previous = reader.last_info
    stamps = dict(reader.last_stamps)
    rig.offset = 100
    calls = []
    def clock():
        calls.append(None)
        if len(calls) == 2:
            raise RuntimeError('failed final SDK clock')
        return 1200
    reader.gdk.Clock.now_ns = clock
    with pytest.raises(RuntimeError, match='final SDK clock'):
        reader.observe()
    assert reader.last_info is previous and reader.last_stamps == stamps


def test_motion_position_must_remain_finite_in_float32_observation(rig):
    reader = rig.make()
    rig.pose.position_m = (1e100, .2, .3)
    with pytest.raises(ValueError, match='pose'):
        reader.observe()


def test_tf_position_error_must_be_finite(rig):
    reader = rig.make()
    rig.tf.poses[1][:3] = [1.7e308, 1.7e308, 1.7e308]
    with pytest.raises(ValueError, match='pose'):
        reader.observe()


@pytest.mark.parametrize('timeout', [0., -1., float('nan'), float('inf'), 31.])
def test_reader_rejects_invalid_timeout_before_gdk_initialization(rig, timeout):
    reader = rig.make()
    initialized = []
    reader.gdk.gdk_init = lambda: initialized.append(True) or 0
    with pytest.raises(ValueError, match='timeout'):
        GdkReader(timeout_s=timeout)
    assert not initialized
