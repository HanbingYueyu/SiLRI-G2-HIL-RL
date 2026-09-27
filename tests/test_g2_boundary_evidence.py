import json
import pytest
from g2_local.local_envelope import LocalEnvelope
from g2_local.real_train import EvalTransitionSink, RunEvidenceWriter
from test_g2_motion_backend import Reader, Port, backend


@pytest.mark.parametrize('stage', ['pre_command_target', 'successor_read'])
def test_boundary_rejection_preserves_pose_reference_and_stage(stage):
    reader = Reader()
    port = Port(reader)
    driver = backend(reader, port, local_envelope=LocalEnvelope((-.01,)*3, (.01,)*3, .2))
    reference = tuple(reader.state)
    driver.episode_reference = reference
    pose = list(reference)
    pose[0] += .02
    try:
        with pytest.raises(ValueError) as caught:
            driver._check_local(pose, stage)
        event = caught.value.boundary_event
        assert event['stage'] == stage
        assert event['pose'] == pose
        assert event['episode_reference'] == list(reference)
        assert 'translation envelope' in event['reason']
        assert not port.sent
    finally:
        driver.close()


def test_roundoff_not_counted_as_action_clipping(tmp_path):
    tmp_path.chmod(0o700)
    evidence = RunEvidenceWriter(tmp_path, tmp_path/'manifest.json', role='eval')
    sink = EvalTransitionSink('run', 'hash', 0, {}, evidence)
    sink.send_transition_batch([dict(done=True, truncated=False, complementary_info=dict(
        episode_id='ep', step_id=0, is_intervention=False, success_label=True,
        selected_action=(.5,)*6, executed_action=(.5+1e-12,)*6))])
    row = json.loads((tmp_path/'episode_summaries.jsonl').read_text())
    assert row['action_clipping_count'] == 0
    assert row['action_numerical_difference_count'] == 1


@pytest.mark.parametrize('after_send', [False, True])
def test_execute_boundary_failure_retains_mapping_and_send_state(after_send):
    reader = Reader()
    reader.state[0] = 0.
    class OvershootPort(Port):
        def send(self, target):
            super().send(target)
            if after_send:
                self.reader.state[0] = .1
    port = OvershootPort(reader)
    limit = .02 if after_send else .005
    driver = backend(reader, port, allow_motion=True,
                     local_envelope=LocalEnvelope((-limit,)*3, (limit,)*3, .2))
    driver.begin_episode(driver.observe())
    try:
        with pytest.raises(ValueError) as caught:
            driver.execute((1., 0., 0., 0., 0., 0.))
        event = caught.value.boundary_event
        assert event['stage'] == ('successor_read' if after_send else 'pre_command_target')
        execution = caught.value.execution_event
        assert execution['send_status'] == ('acknowledged' if after_send else 'not_submitted')
        assert execution['action_mapping']['effective_action'][0] == pytest.approx(1.)
        assert (execution['command_sequence'] is not None) == after_send
        assert bool(port.sent) == after_send
    finally:
        driver.close()


def test_real_workspace_clipping_is_recorded_by_execute():
    reader = Reader()  # x=.995, bound=1., proposed increment=.01
    driver = backend(reader, Port(reader), allow_motion=True)
    try:
        driver.execute((1., 0., 0., 0., 0., 0.))
        event = driver.last_execution_timing['action_mapping']
        assert event['workspace_clip_m'] == pytest.approx([.005, 0., 0.])
        assert event['effective_action'][0] == pytest.approx(.5)
    finally:
        driver.close()


def test_reset_absolute_workspace_failure_has_boundary_event():
    reader = Reader()
    reader.state[0] = 1.1
    port = Port(reader)
    driver = backend(reader, port, local_envelope=LocalEnvelope((-.01,)*3, (.01,)*3, .2))
    try:
        with pytest.raises(ValueError) as caught:
            driver.begin_episode(driver.observe())
        assert caught.value.boundary_event['boundary_kind'] == 'absolute_workspace'
        assert caught.value.boundary_event['stage'] == 'episode_reset'
        assert caught.value.boundary_event['pose'][0] == 1.1
        assert not port.sent
    finally:
        driver.close()


def test_actor_error_identity_survives_coordinator_clear(tmp_path):
    from test_g2_real_actor import actor_rig
    from g2_local.real_train import record_actor_failure
    from types import SimpleNamespace
    rig = actor_rig()
    error = ValueError('boundary failure')
    error.boundary_event = {'stage': 'successor_read'}
    def fail():
        rig.coordinator.context = None
        raise error
    rig.env.during_step = fail
    with pytest.raises(ValueError) as caught:
        rig.runtime.run(max_completed_steps=1)
    assert caught.value.episode_id is not None
    assert caught.value.step_id == 0
    evidence = RunEvidenceWriter(tmp_path, tmp_path/'manifest', role='actor')
    transport = SimpleNamespace(tracker=SimpleNamespace(finalize_unfinished=lambda reason: None))
    record_actor_failure(rig.runtime, transport, evidence, 'actor', error)
    records = [json.loads(line) for line in evidence.events_path.read_text().splitlines()]
    record = next(row for row in records if row['event'] == 'local_envelope_rejected')
    assert record['episode_id'] == error.episode_id


def test_send_timeout_is_unconfirmed_not_unsent():
    import time
    reader = Reader()
    class SlowPort(Port):
        def send(self, target):
            super().send(target)
            time.sleep(.15)  # Real CommandStream watchdog exceeds .1 s lease.
    port = SlowPort(reader)
    driver = backend(reader, port, allow_motion=True)
    try:
        with pytest.raises(RuntimeError) as caught:
            driver.execute((1., 0., 0., 0., 0., 0.))
        event = caught.value.execution_event
        assert event['send_status'] == 'submitted_unconfirmed'
        assert event['command_sequence'] == 1
        assert event['command_sent_monotonic_ns'] is None
        assert event['action_mapping']['workspace_clip_m'][0] == pytest.approx(.005)
        assert port.sent  # An unconfirmed SDK call may already have moved hardware.
    finally:
        driver.close()


def test_actor_reset_failure_keeps_episode_identity(tmp_path):
    from test_g2_real_actor import actor_rig
    rig = actor_rig()
    error = ValueError('Measured pose already outside configured workspace')
    error.boundary_event = {'stage': 'episode_reset', 'boundary_kind': 'absolute_workspace'}
    def fail_reset(**kwargs):
        rig.coordinator.context = None
        raise error
    rig.env.reset = fail_reset
    with pytest.raises(ValueError) as caught:
        rig.runtime.run(max_completed_steps=1)
    assert caught.value.episode_id is not None
    assert caught.value.step_id is None
    assert rig.transport.sent == []
