from dataclasses import replace
from pathlib import Path

import pytest

from test_g2_real_learner import _learner, _row
from test_g2_motion_backend import Reader, Port, backend


def test_beta_requires_complete_imports_and_retained_sample_floor():
    learner = _learner()
    learner.imported_demo_episodes.clear()
    learner.config.optimization = replace(learner.config.optimization,
        beta_min_demo_episodes=2, beta_min_human_transitions=2)
    learner.ingest([_row(human=True)])
    assert learner.pretrain_behavior() is False
    learner.ingest([_row(1, human=True)])
    assert learner.pretrain_behavior() is False
    learner.imported_demo_episodes.update({'demo-a', 'demo-b'})
    assert learner.pretrain_behavior() is True


@pytest.mark.parametrize('rotation', [False, True])
def test_large_policy_pose_drift_never_submits(rotation):
    reader = Reader()
    reader.state[0] = .5
    port = Port(reader)
    driver = backend(reader, port, allow_motion=True)
    try:
        predecessor = driver.observe()
        if rotation:
            from scipy.spatial.transform import Rotation
            reader.state[3:] = Rotation.from_rotvec([0, 0, .03]).as_quat()
        else:
            reader.state[0] += .02
        with pytest.raises(RuntimeError, match='Policy pose drift'):
            driver.execute_from((1, 0, 0, 0, 0, 0), predecessor)
        assert port.sent == []
    finally:
        driver.close()


def test_site_start_lift_is_rejected_without_enlarging_workspace():
    from g2_local.training_config import load_training_config
    from g2_local.auto_reset import reset_waypoints
    config = load_training_config(Path('configs/g2_site_candidate.json'), cli_allow_motion=False)
    pose = (.546524, .175063, .918864, 0., 0., 0., 1.)
    with pytest.raises(ValueError):
        reset_waypoints(pose, pose, config.motion)


def test_identity_detects_env_source_change(tmp_path):
    from g2_local.code_identity import source_digest
    path = tmp_path / 'rl_envs' / 'environment.py'
    path.parent.mkdir()
    path.write_text('reward = 1\n')
    before = source_digest(tmp_path)
    path.write_text('reward = 2\n')
    assert source_digest(tmp_path) != before


def test_pending_sample_is_identified_on_next_step_failure():
    from test_g2_real_actor import actor_rig
    rig = actor_rig()
    events = []
    rig.runtime.telemetry = lambda kind, **fields: events.append((kind, fields))
    def fail_second():
        if rig.env.step_calls == 2:
            raise RuntimeError('next-step-failed')
    rig.env.during_step = fail_second
    with pytest.raises(RuntimeError, match='next-step-failed'):
        rig.runtime.run()
    discarded = [value for kind, value in events if kind == 'pending_transition_discarded']
    assert len(discarded) == 1
    assert discarded[0]['transition_id'] == 'run-1/episode-1/0'
    assert discarded[0]['count'] == 1
    assert not rig.transport.sent
    assert rig.env.closed
