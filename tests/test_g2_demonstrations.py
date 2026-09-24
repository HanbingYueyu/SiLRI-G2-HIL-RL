"""Focused human-only collection -> durable trajectory -> real replay checks."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from g2_local.demonstrations import DemonstrationWriter, import_demonstrations, load_demo_episodes
from g2_local.real_actor import RealActorRuntime
from g2_local.real_learner import RealLearnerRuntime, load_checkpoint
from g2_local.real_train import main
from g2_local.spacemouse import DemonstrationIntervention
from test_g2_intervention import Reader, explicit_config, frame
from test_g2_real_actor import FakeCoordinator, FakeEnv, FakeTransport
from test_g2_real_learner import _config, _row
from g2_local.contract import EpisodeContext


TEMPLATE = Path(__file__).resolve().parents[1] / 'configs/g2_real_training_readonly.json'


def config():
    result = _config()
    result.canonical_payload = json.loads(TEMPLATE.read_text())
    result.task = SimpleNamespace(max_episode_steps=80, success_reward=10.,
                                  failure_reward=-1., step_reward=-.05)
    return result


def demo_row(step=0, terminal=False):
    row = _row(step, human=True, executed=.4)
    row['complementary_info'].update(policy_action=(0.,)*6,
        human_action=(.4,0.,0.,0.,0.,0.), selected_action=(.4,0.,0.,0.,0.,0.),
        reward_source='human', success_label=True if terminal else None)
    row['reward'] = 10. if terminal else -.05
    row['done'] = terminal
    row['next_state']['observation.images.left_wrist'].fill_(.25)
    return row


class Evidence:
    def __init__(self):
        self.events = []
    def event(self, name, **values):
        self.events.append((name, values))


def test_complete_demo_imports_human_actions_pixels_and_resumes_without_duplicates(tmp_path):
    torch.set_num_threads(2)
    cfg = config()
    path = tmp_path / 'demo'
    writer = DemonstrationWriter(path, config=cfg, run_id='run-1')
    writer.append(demo_row())
    with pytest.raises(ValueError, match='No complete'):
        list(load_demo_episodes(path, config=cfg))
    writer.append(demo_row(1, terminal=True))
    learner = RealLearnerRuntime(config=cfg, run_id='training-2')
    evidence = Evidence()
    assert import_demonstrations(learner, [path], evidence=evidence)['accepted'] == 2
    assert len(learner.human_replay) == len(learner.online_replay) == 2
    assert learner.human_replay.actions[0, 0].item() == pytest.approx(.4)
    assert torch.all(learner.human_replay.next_states['observation.images.left_wrist'][0] == .25)
    assert learner.human_replay.dones[:2].tolist() == [False, True]
    assert evidence.events[0][1]['source_run_id'] == 'run-1'
    checkpoint = learner.save_checkpoint(tmp_path / 'checkpoint.pt')
    restored = load_checkpoint(checkpoint, expected_run_id='training-2',
                               expected_config_hash=cfg.config_hash).runtime
    restored.config = cfg
    budget = restored.snapshot_counts()['budget']
    result = import_demonstrations(restored, [path], evidence=evidence)
    assert result['duplicates'] == 2 and result['accepted'] == 0
    assert restored.snapshot_counts()['budget'] == budget


def test_corrupt_or_incompatible_demo_rejected_before_replay_mutation(tmp_path):
    cfg = config()
    path = tmp_path / 'demo'
    writer = DemonstrationWriter(path, config=cfg, run_id='run-1')
    writer.append(demo_row(terminal=True))
    changed = deepcopy(cfg)
    changed.canonical_payload['task']['action_scale'][0] *= 2
    with pytest.raises(ValueError, match='contract mismatch'):
        list(load_demo_episodes(path, config=changed))
    row_path = next(path.glob('*/00000000.pt'))
    with row_path.open('ab') as stream:
        stream.write(b'corruption')
    learner = RealLearnerRuntime(config=cfg, run_id='training-2')
    with pytest.raises(ValueError, match='checksum'):
        import_demonstrations(learner, [path], evidence=Evidence())
    assert len(learner.online_replay) == 0


def test_demo_neutral_keeps_human_authority_and_disconnect_raises():
    reader, now = Reader(), [1.]
    source = DemonstrationIntervention(reader, explicit_config(), clock=lambda: now[0])
    assert source() == (True, (0.,)*6)
    reader.frame = frame(axes=(.15,0.,0.,0.,0.,0.), stamps=(1.1,1.1))
    now[0] = 1.1
    assert 0 < source()[1][0] < .12  # Capture input below automatic takeover threshold.
    reader.frame = frame(stamps=(1.2,1.2)); now[0] = 1.2
    assert source() == (True, (0.,)*6)
    now[0] = 10.
    assert source() == (True, (0.,)*6)
    def disconnected():
        raise OSError('unplugged')
    reader.poll = disconnected
    with pytest.raises(OSError):
        source()


def test_demo_runtime_never_constructs_or_requests_policy(monkeypatch):
    import g2_local.real_actor as module
    monkeypatch.setattr(module, 'create_policy', lambda *_: pytest.fail('policy allocation'))
    coordinator = FakeCoordinator()
    env = FakeEnv(coordinator)
    transport = FakeTransport([])
    transport.receive_latest_parameters = lambda: pytest.fail('parameter request')
    cfg = config()
    runtime = RealActorRuntime(config=cfg, run_id='run-1', config_hash=cfg.config_hash,
        coordinator=coordinator, transport=transport, demonstration=True,
        context_source=SimpleNamespace(read_new=lambda: EpisodeContext(
            'ep-1', (0.,0.,0.), 'test', 'test', visual_reset_monotonic_ns=1)),
        env_factory=lambda *_: env)
    runtime.run(max_completed_steps=1)
    row = transport.sent[0]
    assert row['complementary_info']['policy_action'] == (0.,)*6
    assert row['complementary_info']['is_intervention'] is True
    assert runtime.policy is None


def test_demo_cli_cannot_bypass_motion_gate(tmp_path, monkeypatch):
    import g2_local.real_train as cli
    monkeypatch.setattr(cli, 'import_gdk_runtime', lambda: pytest.fail('hardware imported'))
    assert main(['demo', '--run-id', 'demo-1', '--config', str(TEMPLATE),
                 '--output', str(tmp_path / 'run'), '--allow-motion']) == 2


def test_demo_writer_rejects_policy_rows_and_step_gaps(tmp_path):
    writer = DemonstrationWriter(tmp_path / 'demo', config=config(), run_id='run-1')
    with pytest.raises(ValueError, match='exclusively human'):
        writer.append(_row())
    with pytest.raises(ValueError, match='Non-contiguous'):
        writer.append(demo_row(1))


def test_offline_validation_cli_reports_complete_episode(tmp_path, capsys):
    from g2_local.demonstrations import main as validate
    from g2_local.training_config import load_training_config
    cfg = load_training_config(TEMPLATE, cli_allow_motion=False)
    row = demo_row(terminal=True)
    row['complementary_info']['config_hash'] = cfg.config_hash
    path = tmp_path / 'demo'
    DemonstrationWriter(path, config=cfg, run_id='run-1').append(row)
    assert validate(['--config', str(TEMPLATE), '--dataset', str(path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary['transitions'] == summary['episodes'] == summary['successes'] == 1
    assert summary['motion_authorized'] is False
