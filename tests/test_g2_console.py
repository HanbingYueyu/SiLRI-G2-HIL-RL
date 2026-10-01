"""Operator console line formatting and Learner stdout mirror."""

from g2_local.console import RateLimit, number, say
from g2_local.real_train import LearnerConsole


def test_say_prefixes_a_timestamp(capsys):
    say('hello')
    out = capsys.readouterr().out
    assert out.startswith('[') and out.endswith('hello\n')


def test_number_never_raises_on_missing_or_bad_values():
    assert number(None) == '—'
    assert number('x') == '—'
    assert number(1.23456) == '1.2346'
    assert number(float('nan')) == 'nan'


def test_rate_limit_uses_the_injected_clock():
    now = [0.]
    limit = RateLimit(1., monotonic=lambda: now[0])
    assert limit.due() is True
    assert limit.due() is False
    now[0] += 1.
    assert limit.due() is True


def test_learner_console_reports_losses_updates_and_episode_steps(capsys):
    console = LearnerConsole(interval_s=0.)
    console(dict(event='training_resume', completed_episodes=0, learner_update=10,
                 next_episode=1, checkpoint_source='seed.pt'))
    console(dict(event='learner_update', learner_update=11, critic_loss=.5,
                 actor_loss=None, completed_episodes=0))
    console(dict(event='learner_update', learner_update=12, critic_loss=.25,
                 actor_loss=-.125, completed_episodes=0))
    console(dict(event='episode_completed', completed_episodes=1, next_episode=2,
                 learner_update=13, episode_id='ep-1', steps=187))
    console(dict(event='checkpoint_saved', completed_episodes=1, learner_update=13,
                 path='runtime/fixed/checkpoint.pt'))
    out = capsys.readouterr().out
    assert '训练已恢复：已完成回合 0' in out
    assert 'critic 0.5000' in out and 'actor —' in out
    assert '更新 12｜critic 0.2500｜actor -0.1250' in out
    assert '回合完成：第 1 个，本回合 187 步' in out
    assert '检查点已保存：runtime/fixed/checkpoint.pt' in out
