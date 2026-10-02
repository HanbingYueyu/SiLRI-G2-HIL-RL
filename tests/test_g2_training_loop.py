"""Loop-mode launcher plumbing: Section 6 between episodes, stop-on-error.

The launcher is an operator script under ``scripts/``; these tests load it
directly and replace only its child-process boundary, so the ordering, the exit
code rules and the Actor command line are covered without hardware.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'g2_start_training_actor', ROOT / 'scripts/start_training_actor.py')
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def test_loop_runs_section_six_between_episodes_and_stops_on_actor_error(monkeypatch):
    calls = []
    codes = iter([0, 130, 1])
    monkeypatch.setattr(launcher, 'spacemouse_holder', lambda path: None)
    monkeypatch.setattr(launcher, 'write_context',
                        lambda root, path, run_id: (Path(path), {'episode_id': 'ep-1'}))
    monkeypatch.setattr(launcher, 'run_section_six',
                        lambda root: calls.append('section-six') or 0)
    monkeypatch.setattr(launcher, 'run_pre_reset',
                        lambda root, config_path: calls.append('pre-reset') or 0)

    def fake_episode(args, root, config_path, *, index):
        calls.append(f'actor-{index}')
        return next(codes)

    monkeypatch.setattr(launcher, 'run_one_episode', fake_episode)
    code = launcher.main(['--allow-motion', '--loop', '5'])
    # 0 = Y/F clean exit continues; 130 = operator Ctrl+C continues; anything else stops.
    assert code == 1
    # Episode 1 starts from the operator's already-reset scene, so the retraction
    # only happens between episodes — mirroring the demonstration sequence.
    assert calls == ['section-six', 'actor-1',
                     'pre-reset', 'section-six', 'actor-2',
                     'pre-reset', 'section-six', 'actor-3']


def test_retraction_stops_the_loop_when_it_fails(monkeypatch):
    calls = []
    monkeypatch.setattr(launcher, 'spacemouse_holder', lambda path: None)
    monkeypatch.setattr(launcher, 'write_context',
                        lambda root, path, run_id: (Path(path), {'episode_id': 'ep-1'}))
    monkeypatch.setattr(launcher, 'run_section_six',
                        lambda root: calls.append('section-six') or 0)
    monkeypatch.setattr(launcher, 'run_pre_reset',
                        lambda root, config_path: calls.append('pre-reset') or 4)
    monkeypatch.setattr(launcher, 'run_one_episode',
                        lambda args, root, config_path, *, index: calls.append(f'actor-{index}') or 0)
    assert launcher.main(['--allow-motion', '--loop', '3', '--skip-first-reset']) == 4
    assert calls == ['actor-1', 'pre-reset']


def test_skip_pre_reset_drops_only_the_retraction(monkeypatch):
    calls = []
    monkeypatch.setattr(launcher, 'spacemouse_holder', lambda path: None)
    monkeypatch.setattr(launcher, 'write_context',
                        lambda root, path, run_id: (Path(path), {'episode_id': 'ep-1'}))
    monkeypatch.setattr(launcher, 'run_section_six',
                        lambda root: calls.append('section-six') or 0)
    monkeypatch.setattr(launcher, 'run_pre_reset',
                        lambda root, config_path: calls.append('pre-reset') or 0)
    monkeypatch.setattr(launcher, 'run_one_episode',
                        lambda args, root, config_path, *, index: calls.append(f'actor-{index}') or 0)
    assert launcher.main(['--allow-motion', '--loop', '2',
                          '--skip-pre-reset']) == 0
    assert calls == ['section-six', 'actor-1', 'section-six', 'actor-2']


def test_loop_stops_when_section_six_fails(monkeypatch):
    calls = []
    monkeypatch.setattr(launcher, 'spacemouse_holder', lambda path: None)
    monkeypatch.setattr(launcher, 'write_context',
                        lambda root, path, run_id: (Path(path), {'episode_id': 'ep-1'}))
    monkeypatch.setattr(launcher, 'run_section_six',
                        lambda root: calls.append('section-six') or 3)
    monkeypatch.setattr(launcher, 'run_one_episode',
                        lambda *a, **k: calls.append('actor') or 0)
    assert launcher.main(['--allow-motion', '--loop', '5']) == 3
    assert calls == ['section-six']


def test_loop_mode_makes_the_actor_exit_after_yf():
    root = ROOT
    loop_args = launcher.parser().parse_args(['--allow-motion', '--loop', '2'])
    loop_cmd = launcher.actor_command(loop_args, root, root / 'c.json', root / 'o',
                                      root / 'ctx.json')
    assert '--exit-after-labeled-demo' in loop_cmd
    single = launcher.parser().parse_args(['--allow-motion'])
    single_cmd = launcher.actor_command(single, root, root / 'c.json', root / 'o',
                                        root / 'ctx.json')
    assert '--exit-after-labeled-demo' not in single_cmd
    assert '--auto-reset' not in loop_cmd


def test_busy_spacemouse_is_reported_before_writing_a_context(monkeypatch, capsys):
    written = []
    monkeypatch.setattr(launcher, 'spacemouse_holder', lambda path: 'busy')
    monkeypatch.setattr(launcher, 'run_section_six', lambda root: pytest.fail('Section 6 ran'))
    monkeypatch.setattr(launcher, 'write_context',
                        lambda root, path, run_id: written.append(path))
    assert launcher.main(['--allow-motion']) == 2
    assert written == []
    assert '已被占用' in capsys.readouterr().out


def test_single_episode_skips_section_six_when_the_operator_already_ran_it(monkeypatch):
    """The manual (non-loop) flow runs Section 6 by hand, then starts the Actor;
    without --skip-first-reset the script would run Section 6 a second time."""
    calls = []
    monkeypatch.setattr(launcher, 'spacemouse_holder', lambda path: None)
    monkeypatch.setattr(launcher, 'write_context',
                        lambda root, path, run_id: (Path(path), {'episode_id': 'ep-1'}))
    monkeypatch.setattr(launcher, 'run_section_six',
                        lambda root: calls.append('section-six') or 0)
    monkeypatch.setattr(launcher, 'run_one_episode',
                        lambda args, root, config_path, *, index: calls.append(f'actor-{index}') or 0)
    assert launcher.main(['--allow-motion', '--skip-first-reset']) == 0
    assert calls == ['actor-1']
    calls.clear()
    assert launcher.main(['--allow-motion']) == 0
    assert calls == ['section-six', 'actor-1']


def test_an_unpatched_retraction_seam_cannot_reach_the_robot(monkeypatch):
    """Regression: a loop test once left ``run_pre_reset`` unpatched, so the real
    pre-reset was spawned and retracted the physical arm. The session safety net
    (tests/conftest.py) must now make that fail loudly instead.
    """
    monkeypatch.setattr(launcher, 'spacemouse_holder', lambda path: None)
    monkeypatch.setattr(launcher, 'write_context',
                        lambda root, path, run_id: (Path(path), {'episode_id': 'ep-1'}))
    monkeypatch.setattr(launcher, 'run_section_six', lambda root: 0)
    monkeypatch.setattr(launcher, 'run_one_episode', lambda *a, **k: 0)
    # run_pre_reset is deliberately NOT patched.
    with pytest.raises(RuntimeError, match='Refusing to spawn'):
        launcher.main(['--allow-motion', '--loop', '2', '--skip-first-reset'])


def test_skip_first_reset_skips_section_six_only_for_episode_one(monkeypatch):
    calls = []
    monkeypatch.setattr(launcher, 'spacemouse_holder', lambda path: None)
    monkeypatch.setattr(launcher, 'write_context',
                        lambda root, path, run_id: (Path(path), {'episode_id': 'ep-1'}))
    monkeypatch.setattr(launcher, 'run_section_six',
                        lambda root: calls.append('section-six') or 0)
    monkeypatch.setattr(launcher, 'run_pre_reset',
                        lambda root, config_path: calls.append('pre-reset') or 0)
    monkeypatch.setattr(launcher, 'run_one_episode',
                        lambda args, root, config_path, *, index: calls.append(f'actor-{index}') or 0)
    assert launcher.main(['--allow-motion', '--loop', '2', '--skip-first-reset']) == 0
    # Episode 1 skipped Section 6, and still no retraction before it (nothing ran yet).
    assert calls == ['actor-1', 'pre-reset', 'section-six', 'actor-2']


def test_loop_bound_and_write_context_conflicts_are_rejected(monkeypatch):
    monkeypatch.setattr(launcher, 'spacemouse_holder', lambda path: None)
    with pytest.raises(SystemExit):
        launcher.main(['--allow-motion', '--loop', '201'])
    with pytest.raises(SystemExit):
        launcher.main(['--write-context', '--loop', '2'])
    with pytest.raises(SystemExit):
        launcher.main(['--write-context', '--allow-motion'])
