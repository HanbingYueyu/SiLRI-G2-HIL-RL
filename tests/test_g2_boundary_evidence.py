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
