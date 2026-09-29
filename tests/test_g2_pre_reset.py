from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from g2_local.config import LocalTaskConfig
from g2_local.local_envelope import LocalEnvelope
from g2_local.pre_reset import pre_reset_waypoints, reset_only_config, run_pre_reset


def motion_config(*, y_high=.30):
    limits = LocalTaskConfig(
        action_scale=(.01, .01, .01, .02, .02, .02),
        workspace_low=(.4, .0, .8), workspace_high=(.7, y_high, 1.0))
    return SimpleNamespace(
        limits=limits,
        local_envelope=LocalEnvelope((-.15, -.15, -.15), (.15, .15, .15), .4),
        auto_reset=SimpleNamespace(linear_speed_m_s=.01,
                                   position_tolerance_m=.001, timeout_s=30.),
        policy_rotation_drift_rad=.02)


def test_reset_only_workspace_allows_approved_path_without_changing_demo_limits():
    @dataclass(frozen=True)
    class Reset:
        linear_speed_m_s: float = .01

    @dataclass(frozen=True)
    class Motion:
        limits: LocalTaskConfig
        auto_reset: Reset

    @dataclass(frozen=True)
    class Config:
        motion: Motion

    original = Config(Motion(LocalTaskConfig(
        action_scale=(.0015,)*3 + (.026,)*3,
        workspace_low=(.4765, .0901, .8289),
        workspace_high=(.6665, .2451, .965)), Reset()))
    reset = reset_only_config(original)
    start = (.5847077, .17285539, .94144654, 0., 0., 0., 1.)

    with pytest.raises(ValueError, match='absolute workspace'):
        pre_reset_waypoints(start, SimpleNamespace(
            limits=original.motion.limits, local_envelope=None))
    lift, shift = pre_reset_waypoints(start, SimpleNamespace(
        limits=reset.motion.limits, local_envelope=None))
    assert lift[2] == pytest.approx(.99144654)
    assert shift[1] == pytest.approx(.27285539)
    assert reset.motion.limits.workspace_high == (.6665, .28, 1.00)
    assert original.motion.limits.workspace_high == (.6665, .2451, .965)
    assert reset.motion.auto_reset.linear_speed_m_s == .04
    assert original.motion.auto_reset.linear_speed_m_s == .01


def test_pre_reset_main_passes_expanded_limits_only_to_its_own_backend(monkeypatch):
    @dataclass(frozen=True)
    class Reset:
        linear_speed_m_s: float = .01

    from g2_local import pre_reset

    @dataclass(frozen=True)
    class Motion:
        limits: LocalTaskConfig
        auto_reset: Reset

    @dataclass(frozen=True)
    class Config:
        motion: Motion
        motion_permitted: bool = True

    original = Config(Motion(LocalTaskConfig(
        action_scale=(.0015,)*3 + (.026,)*3,
        workspace_low=(.4765, .0901, .8289),
        workspace_high=(.6665, .2451, .965)), Reset()))
    seen = {}
    monkeypatch.setattr(pre_reset, 'load_training_config',
                        lambda *args, **kwargs: original)

    def create(config, coordinator, *, cli_allow_motion, skip_tf_progress=False):
        seen['backend_config'] = config
        assert cli_allow_motion is True
        assert skip_tf_progress is True
        return object()

    def run(env, motion, *, start_limits):
        seen['reset_motion'] = motion
        seen['start_limits'] = start_limits

    monkeypatch.setattr(pre_reset, 'create_motion_env', create)
    monkeypatch.setattr(pre_reset, 'run_pre_reset', run)
    assert pre_reset.main(['--config', 'unused.json', '--allow-motion']) == 0
    assert seen['backend_config'].motion.limits.workspace_high == (.6665, .28, 1.0)
    assert seen['reset_motion'] is seen['backend_config'].motion
    assert seen['start_limits'] is original.motion.limits
    assert original.motion.limits.workspace_high == (.6665, .2451, .965)
    assert seen['reset_motion'].auto_reset.linear_speed_m_s == .04
    assert original.motion.auto_reset.linear_speed_m_s == .01


def test_pre_reset_corrects_orientation_drift_without_aborting_translation():
    motion = motion_config()
    state = np.array([.55, .12, .9, 0., 0., 0., 1.], dtype=np.float64)
    actions = []

    class Backend:
        step_period = 1.

        def observe(self):
            return {'state': state.copy()}

        def begin_episode(self, observation):
            pass

        def execute(self, action):
            actions.append(np.asarray(action).copy())
            state[:3] += np.asarray(action[:3]) * motion.limits.action_scale[:3]
            state[3:] = (Rotation.from_rotvec(
                np.asarray(action[3:]) * motion.limits.action_scale[3:]) *
                Rotation.from_quat(state[3:])).as_quat()
            if len(actions) == 1:
                state[3:] = Rotation.from_euler('z', .03).as_quat()
            return SimpleNamespace(observation={'state': state.copy()})

    class Env:
        backend = Backend()

        def close(self):
            pass

    run_pre_reset(Env(), motion, emit=lambda *_: None)
    assert state[:3] == pytest.approx((.55, .22, .95))
    assert np.linalg.norm(Rotation.from_quat(state[3:]).as_rotvec()) < 1e-6
    assert any(np.linalg.norm(action[3:]) > 0 for action in actions[1:])


def test_pre_reset_waypoints_are_base_z_then_base_y_and_keep_orientation():
    motion = motion_config()
    current = (.55, .12, .9, 0., 0., 0., 1.)

    lift, shift = pre_reset_waypoints(current, motion)

    assert lift == pytest.approx((.55, .12, .95, 0., 0., 0., 1.))
    assert shift == pytest.approx((.55, .22, .95, 0., 0., 0., 1.))


def test_pre_reset_prechecks_both_targets_before_any_motion():
    motion = motion_config(y_high=.20)
    current = (.55, .15, .9, 0., 0., 0., 1.)
    with pytest.raises(ValueError, match='absolute workspace'):
        pre_reset_waypoints(current, motion)


def test_run_pre_reset_executes_z_then_y_at_configured_speed_and_closes():
    motion = motion_config()
    state = np.array([.55, .12, .9, 0., 0., 0., 1.], dtype=np.float64)
    actions = []

    class Backend:
        step_period = 1.

        def observe(self):
            return {'state': state.copy()}

        def begin_episode(self, observation):
            assert np.array_equal(observation['state'], state)

        def execute(self, action):
            actions.append(np.asarray(action).copy())
            state[:3] += np.asarray(action[:3]) * np.asarray(motion.limits.action_scale[:3])
            return SimpleNamespace(observation={'state': state.copy()})

    class Env:
        backend = Backend()
        closed = False

        def close(self):
            self.closed = True

    env = Env()
    messages = []
    run_pre_reset(env, motion, emit=messages.append)

    assert env.closed
    assert state[:3] == pytest.approx((.55, .22, .95))
    axes = [int(np.argmax(np.abs(action[:3]))) for action in actions]
    assert axes == [2] * 5 + [1] * 10
    assert all(action[3:] == pytest.approx((0., 0., 0.)) for action in actions)
    assert all(np.max(np.abs(action[:3])) <= 1. for action in actions)
    assert any('+Z' in message for message in messages)
    assert any('+Y' in message for message in messages)


def test_run_pre_reset_closes_and_sends_no_command_when_path_is_out_of_bounds():
    motion = motion_config(y_high=.20)
    state = np.array([.55, .15, .9, 0., 0., 0., 1.], dtype=np.float64)

    class Backend:
        step_period = 1.
        commands = 0

        def observe(self):
            return {'state': state.copy()}

        def begin_episode(self, observation):
            raise AssertionError('Must reject path before latching for motion')

        def execute(self, action):
            self.commands += 1
            raise AssertionError('Out-of-bounds path must not move')

    class Env:
        backend = Backend()
        closed = False

        def close(self):
            self.closed = True

    env = Env()
    with pytest.raises(ValueError, match='absolute workspace'):
        run_pre_reset(env, motion, emit=lambda *_: None)
    assert env.closed
    assert env.backend.commands == 0


def test_reset_only_range_does_not_admit_a_start_outside_demo_workspace():
    motion = motion_config()
    normal_limits = LocalTaskConfig(
        action_scale=motion.limits.action_scale,
        workspace_low=(.4, .0, .8), workspace_high=(.7, .2, .965))
    state = np.array([.55, .21, .9, 0., 0., 0., 1.])

    class Backend:
        def observe(self):
            return {'state': state.copy()}

        def begin_episode(self, observation):
            raise AssertionError('Outside start must not arm')

    class Env:
        backend = Backend()
        closed = False

        def close(self):
            self.closed = True

    env = Env()
    with pytest.raises(ValueError, match='normal demo workspace'):
        run_pre_reset(env, motion, start_limits=normal_limits)
    assert env.closed
