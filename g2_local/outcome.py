"""Explicit task outcome adapters for the local insertion loop.

An outcome is evaluated only after the driver has accepted an action and
returned a valid successor observation.  The adapter does not inspect or
command hardware; it only turns an operator/classifier label into the reward
and terminal fields carried by a transition.
"""
from dataclasses import dataclass
import math

from .config import HingeInsertTaskConfig


_REWARD_SOURCES = frozenset(('human', 'classifier', 'environment', 'unknown'))


@dataclass(frozen=True)
class OutcomeDecision:
    reward: float
    terminated: bool
    reward_source: str = 'unknown'
    success_label: bool | None = None

    def __post_init__(self):
        if type(self.reward) not in (int, float) or not math.isfinite(self.reward):
            raise ValueError('Outcome reward must be finite')
        if type(self.terminated) is not bool:
            raise ValueError('Outcome terminated must be boolean')
        if (type(self.reward_source) is not str or
                self.reward_source not in _REWARD_SOURCES):
            raise ValueError('Outcome reward_source is invalid')
        if self.success_label is not None and type(self.success_label) is not bool:
            raise ValueError('Outcome success_label must be boolean or None')
        object.__setattr__(self, 'reward', float(self.reward))


def coerce_outcome(value):
    """Accept the legacy ``(reward, terminated)`` callback shape safely."""
    if isinstance(value, OutcomeDecision):
        return value
    if type(value) is tuple and len(value) == 2:
        return OutcomeDecision(value[0], value[1])
    raise ValueError('Outcome callback must return OutcomeDecision or a 2-tuple')


class HumanBinaryOutcome:
    """Map an explicit operator label to the MVP task reward.

    ``label_reader`` returns ``None`` while the episode is ongoing, ``True``
    for a successful insertion, or ``False`` for a terminal failed attempt.
    The callback is deliberately explicit: no image heuristic or hidden
    success assumption is introduced into the training contract.
    """

    def __init__(self, task: HingeInsertTaskConfig, label_reader):
        if type(task) is not HingeInsertTaskConfig:
            raise ValueError('HingeInsertTaskConfig is required')
        if not callable(label_reader):
            raise ValueError('A label reader callback is required')
        self.task = task
        self.label_reader = label_reader

    def __call__(self, observation):
        label = self.label_reader(observation)
        if label is not None and type(label) is not bool:
            raise ValueError('Operator success label must be boolean or None')
        if label is True:
            reward = self.task.success_reward
            terminated = True
        elif label is False:
            reward = self.task.step_reward
            terminated = True
        else:
            reward = self.task.step_reward
            terminated = False
        return OutcomeDecision(reward, terminated, self.task.reward_source, label)
