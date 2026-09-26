from types import SimpleNamespace as NS
import numpy as np
import pytest
from g2_local.config import LocalTaskConfig
from g2_local.auto_reset import AutoResetConfig, reset_waypoints, run_reset


def config():
    return NS(limits=LocalTaskConfig(action_scale=(.0015,)*3+(.026,)*3,
                                   workspace_low=(-1,)*3, workspace_high=(1,)*3),
              local_envelope=None, auto_reset=AutoResetConfig(enabled=True))


def test_lift_first_fixed_orientation_and_workspace_refusal():
    start = (0,0,.5,0,0,0,1)
    end = (.04,0,.43,0,0,0,1)
    points = reset_waypoints(end, start, config())
    assert points[0] == pytest.approx((.04,0,.48,0,0,0,1))
    assert points[1] == start
    with pytest.raises(ValueError, match='workspace'):
        reset_waypoints((0,0,.98,0,0,0,1), start, config())


def test_reset_uses_backend_not_gym_and_cancel_closes():
    from g2_local.motion import plan_target
    cfg = config()
    start = (0,0,.5,0,0,0,1)
    class Backend:
        step_period = .1
        def __init__(self):
            self.pose = (.02,0,.43,0,0,0,1)
            self.trace = []
        def observe(self):
            return {'state': np.array(self.pose)}
        def begin_episode(self, obs):
            self.reference = tuple(obs['state'])
        def execute(self, action):
            self.pose, _ = plan_target(self.pose, action, cfg.limits)
            self.trace.append(self.pose)
            return NS(observation=self.observe())
    class Env:
        def __init__(self):
            self.backend = Backend()
            self.closed = False
        def close(self):
            self.closed = True
    env = Env()
    out = run_reset(env, start, cfg, poll=lambda: None)
    assert env.closed
    assert out['state'][:3] == pytest.approx(start[:3], abs=.001)
    assert all(p[0] == .02 for p in env.backend.trace[:40])
    env = Env()
    def cancel():
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run_reset(env, start, cfg, poll=cancel)
    assert env.closed and not env.backend.trace
