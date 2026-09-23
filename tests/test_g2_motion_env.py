"""Commissioned motion assembly with isolated clock, reader, and port fakes."""
from types import SimpleNamespace as NS
import threading
import time

import numpy as np
import pytest

from g2_local.config import LocalTaskConfig
from g2_local.contract import EpisodeContext
from g2_local.freshness import FreshnessLimits
from g2_local.command_stream import CommandStream
from g2_local.gdk_backend import GdkCommandPort
from g2_local.motion_backend import PoseTarget
from g2_local.motion_env import FreshnessLeaseGuard, MotionFactories, create_motion_env


class Commissioning:
    clock_socket = '/fake/clock.sock'
    expected_master = 'fake-master'

    def __init__(self, events, approved=True):
        self.events = events
        self.approved = approved

    def verify_files_and_hashes(self):
        self.events.append('evidence')
        return self.approved


def config(events, *, permitted=True, approved=True, rois=None):
    limits = LocalTaskConfig(action_scale=(.01,) * 6,
                             workspace_low=(0., 0., .5),
                             workspace_high=(1., 1., 1.))
    motion = NS(limits=limits, control_mode=1, command_timeout_s=.2,
                send_timeout_s=.1, stop_timeout_s=.5, reader_timeout_s=.1,
                command_lifetime_s=.1, send_rate_hz=50., adapter_root='/fake/adapter')
    return NS(requested_motion=True, motion_permitted=permitted,
              commissioning=Commissioning(events, approved), motion=motion,
              freshness=FreshnessLimits(.1, .1, .1, .1, .1, .1),
              task=NS(control_hz=50., max_episode_steps=5),
              observation=NS(image_size=128, camera_rois=rois or {}))


class FakeReader:
    def __init__(self, events, *, mode=1):
        self.events = events
        self.closed = False
        self.state = np.array([.3, .3, .8, 0., 0., 0., 1.], dtype=np.float32)
        self.controller = NS(checked_arm_state=lambda: events.append('arm_state'),
                             motion_status_summary=lambda: {'control_mode': mode,
                                                            'error_code': 0})
        self.last_info = {}

    def observe(self):
        self.last_info = {'captured': time.monotonic()}
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(200, dtype=np.uint8)[None, :]
        image[:, :, 1] = np.arange(200, dtype=np.uint8)[:, None]
        return {'state': self.state.copy(), 'left_wrist': image.copy(),
                'right_aux': image.copy()}

    def close(self):
        assert not self.closed
        self.closed = True
        self.events.append('reader_close')


class FakePort:
    def __init__(self, reader, guard, events):
        self.reader = reader
        self.guard = guard
        self.events = events
        self.sent = 0
        self.stopped = False

    def send(self, target):
        if self.guard() is not True:
            raise RuntimeError('feedback lease expired')
        self.sent += 1
        self.reader.state = np.array((*target.position_m, *target.orientation_xyzw),
                                     dtype=np.float32)

    def stop(self):
        self.stopped = True
        self.events.append('port_stop')


def fake_factories(events, *, mode=1, fail_port=False):
    state = NS(reader=None, clock=None, port=None)

    class Clock:
        def __init__(self):
            self.closed = False

        def read(self):
            raise AssertionError('observation guard is stubbed for assembly tests')

        def close(self):
            assert not self.closed
            self.closed = True
            events.append('clock_close')

    def make_clock(socket, master):
        assert (socket, master) == ('/fake/clock.sock', 'fake-master')
        events.append('clock')
        state.clock = Clock()
        return state.clock

    def make_reader(*, adapter_root, timeout_s, allow_motion):
        assert (adapter_root, timeout_s, allow_motion) == ('/fake/adapter', .1, True)
        events.append('reader')
        state.reader = FakeReader(events, mode=mode)
        return state.reader

    def make_port(controller, *, expected_mode, allow_motion,
                  freshness_guard, life_time_s):
        assert controller is state.reader.controller
        assert (expected_mode, allow_motion, life_time_s) == (1, True, .1)
        events.append('command_port')
        if fail_port:
            raise RuntimeError('port construction failed')
        state.port = FakePort(state.reader, freshness_guard, events)
        return state.port

    return MotionFactories(make_clock, make_reader, make_port), state


def coordinator():
    return NS(intervention=lambda: (False, None), outcome=lambda obs: (0., False))


def test_readonly_config_denies_before_any_resource_or_production_import():
    events = []
    factories, _ = fake_factories(events)
    with pytest.raises(PermissionError):
        create_motion_env(config(events, permitted=False), coordinator(),
                          cli_allow_motion=True, factories=factories)
    assert events == []
    with pytest.raises(PermissionError):
        create_motion_env(config(events), coordinator(),
                          cli_allow_motion=False, factories=factories)
    assert events == []


def test_rechecked_commissioning_evidence_denies_before_clock_or_port():
    events = []
    factories, _ = fake_factories(events)
    with pytest.raises(PermissionError, match='commission'):
        create_motion_env(config(events, approved=False), coordinator(),
                          cli_allow_motion=True, factories=factories)
    assert events == ['evidence']


def test_mode_preflight_denies_and_releases_reader_then_clock_without_port():
    events = []
    factories, state = fake_factories(events, mode=3)
    with pytest.raises(RuntimeError, match='control mode'):
        create_motion_env(config(events), coordinator(),
                          cli_allow_motion=True, factories=factories)
    assert events == ['evidence', 'clock', 'reader', 'arm_state',
                      'reader_close', 'clock_close']
    assert state.reader.closed and state.clock.closed


def test_mode_preflight_rejects_boolean_status_before_command_port():
    events = []
    factories, state = fake_factories(events, mode=True)
    with pytest.raises(RuntimeError, match='control mode'):
        create_motion_env(config(events), coordinator(),
                          cli_allow_motion=True, factories=factories)
    assert 'command_port' not in events
    assert state.reader.closed and state.clock.closed


def test_failed_port_construction_releases_reader_then_clock():
    events = []
    factories, state = fake_factories(events, fail_port=True)
    with pytest.raises(RuntimeError, match='port construction'):
        create_motion_env(config(events), coordinator(),
                          cli_allow_motion=True, factories=factories)
    assert events[-2:] == ['reader_close', 'clock_close']
    assert state.reader.closed and state.clock.closed


def test_assembly_owns_one_session_and_explicit_limits(monkeypatch):
    monkeypatch.setattr('g2_local.motion_env.ObservationFreshnessGuard',
                        lambda client, limits: lambda obs, info, after: True)
    events = []
    factories, state = fake_factories(events)
    env = create_motion_env(config(events), coordinator(),
                            cli_allow_motion=True, factories=factories)
    assert env.backend.enabled is True
    assert tuple(env.backend.config.workspace_low) == (0., 0., .5)
    env.close()
    env.close()
    assert state.reader.closed and state.clock.closed
    assert events.count('reader_close') == events.count('clock_close') == 1
    assert events.index('reader_close') < events.index('clock_close')


def test_feedback_lease_expires_without_another_gym_step(monkeypatch):
    monkeypatch.setattr('g2_local.motion_env.ObservationFreshnessGuard',
                        lambda client, limits: lambda obs, info, after: True)
    events = []
    factories, state = fake_factories(events)
    env = create_motion_env(config(events), coordinator(),
                            cli_allow_motion=True, factories=factories)
    try:
        env.reset(options={'context': EpisodeContext('test', (0., 0., 0.), 'fake', 'fake')})
        env.step(np.zeros(6, dtype=np.float32))
        deadline = time.monotonic() + 1.
        while not env.backend.stream.halt.is_set() and time.monotonic() < deadline:
            time.sleep(.005)
        assert env.backend.stream.halt.is_set()
        assert state.port.sent > 0
        assert state.port.stopped
    finally:
        env.close()


def test_freshness_lease_requires_exact_true_and_expires():
    now = [10.]
    accepted = [True]
    lease = FreshnessLeaseGuard(lambda obs, info, after: accepted[0],
                                feedback_lease_s=.1, clock=lambda: now[0])
    assert lease() is False
    assert lease.accept({}, {}, None) is True
    assert lease() is True
    now[0] = 10.11
    assert lease() is False
    accepted[0] = 1
    assert lease.accept({}, {}, None) == 1
    assert lease() is False


def test_camera_roi_crops_before_resize_and_rejects_out_of_frame(monkeypatch):
    monkeypatch.setattr('g2_local.motion_env.ObservationFreshnessGuard',
                        lambda client, limits: lambda obs, info, after: True)
    events = []
    factories, _ = fake_factories(events)
    rois = {'left_wrist': (10, 20, 100, 80), 'right_aux': (30, 40, 120, 90)}
    env = create_motion_env(config(events, rois=rois), coordinator(),
                            cli_allow_motion=True, factories=factories)
    try:
        obs, _ = env.reset(options={'context': EpisodeContext('test', (0., 0., 0.), 'fake', 'fake')})
        assert obs['left_wrist'].shape == (128, 128, 3)
        assert env.last_crop_boxes == rois
        assert tuple(obs['left_wrist'][0, 0, :2]) == (10, 20)
        assert tuple(obs['right_aux'][0, 0, :2]) == (30, 40)
    finally:
        env.close()

    events = []
    factories, state = fake_factories(events)
    bad = {'left_wrist': (10, 20, 201, 80)}
    env = create_motion_env(config(events, rois=bad), coordinator(),
                            cli_allow_motion=True, factories=factories)
    with pytest.raises(ValueError, match='ROI'):
        env.reset(options={'context': EpisodeContext('test2', (0., 0., 0.), 'fake', 'fake')})
    assert env.backend.stopped
    env.close()
    assert state.reader.closed and state.clock.closed


def test_watchdog_halt_cancels_send_waiting_inside_freshness_preflight():
    entered, release = threading.Event(), threading.Event()
    sends = []

    class Controller:
        def checked_arm_state(self):
            return None

        def motion_status_summary(self):
            return {'control_mode': 1, 'error_code': 0}

        def _send_left_cartesian_pose(self, target, life_time_s):
            sends.append((target, life_time_s))

        def read_end_effector_pose(self, name):
            return PoseTarget((.3, .3, .8), (0., 0., 0., 1.))

    def blocked_freshness():
        entered.set()
        assert release.wait(1.)
        return True

    port = GdkCommandPort(Controller(), expected_mode=1, allow_motion=True,
                          freshness_guard=blocked_freshness, life_time_s=.1)
    stream = CommandStream(port, command_timeout=.3, send_timeout=.05,
                           stop_timeout=.3)
    stream.submit(PoseTarget((.3, .3, .8), (0., 0., 0., 1.)))
    try:
        assert entered.wait(.5)
        assert stream.halt.wait(.5)
        assert sends == []
    finally:
        release.set()
    stop_error = None
    try:
        stream.stop()
    except RuntimeError as error:
        stop_error = error
    assert sends == []
    assert stop_error is not None and 'physical stop unconfirmed' in str(stop_error)


def test_failed_sdk_send_cannot_make_stop_appear_confirmed():
    class FailingController:
        def checked_arm_state(self):
            return None

        def motion_status_summary(self):
            return {'control_mode': 1, 'error_code': 0}

        def _send_left_cartesian_pose(self, target, life_time_s):
            raise OSError('SDK send failed')

    port = GdkCommandPort(FailingController(), expected_mode=1, allow_motion=True,
                          freshness_guard=lambda: True, life_time_s=.1)
    stream = CommandStream(port, command_timeout=.3, send_timeout=.1)
    sequence = stream.submit(PoseTarget((.3, .3, .8), (0., 0., 0., 1.)))
    with pytest.raises(RuntimeError, match='SDK send failed'):
        stream.wait_sent(sequence, timeout=.3)
    with pytest.raises(RuntimeError, match='physical stop unconfirmed'):
        stream.stop()
    assert stream.stop_fault is not None
