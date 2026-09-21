"""Episode lifecycle core; hardware and Gym bindings are separate integrations."""
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Protocol, Any
import logging

from .contract import EpisodeContext, select_action, transition


@dataclass(frozen=True)
class StepResult:
    observation: Any
    executed_action: tuple[float, ...]
    reward: float
    terminated: bool


class Backend(Protocol):
    """Implementations must validate live observations and command acceptance.

    execute returns only after acquiring a valid post-command observation.
    stop must implement the documented backend stop/hold semantics.
    This protocol alone does not constitute a GDK driver.
    """
    def observe(self) -> Any: ...
    def execute(self, action: tuple[float, ...]) -> StepResult: ...
    def stop(self) -> None: ...


class EpisodeRunner:
    def __init__(self, backend: Backend, *, max_steps: int):
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError('max_steps must be a positive integer')
        self.backend = backend
        self.max_steps = max_steps
        self.active = False
        self.context = None
        self.observation = None
        self.steps = 0
        self.assisted = False

    def reset(self, context: EpisodeContext):
        """Begin after operator/upstream scene reset; never execute a reset trajectory."""
        if not isinstance(context, EpisodeContext):
            raise TypeError('EpisodeContext required')
        was_active = self.active
        self.active = False
        if was_active:
            self.backend.stop()
        observation = self.backend.observe()
        if observation is None:
            raise ValueError('Missing initial observation')
        self.observation = deepcopy(observation)
        self.context = context
        self.steps = 0
        self.assisted = False
        self.active = True
        return deepcopy(observation)

    def step(self, policy, *, human_active=False, human=None):
        if not self.active:
            raise RuntimeError('Reset required before stepping')
        try:
            decision = select_action(policy, human_active=human_active, human=human)
            result = self.backend.execute(decision.selected_action)
            truncated = self.steps + 1 >= self.max_steps and not result.terminated
            row = transition(self.observation, result.observation, decision,
                             result.executed_action, reward=result.reward,
                             terminated=result.terminated, truncated=truncated)
            self.assisted = self.assisted or decision.is_intervention
            row['complementary_info'].update(asdict(self.context))
            row['complementary_info'].update(
                step_id=self.steps, episode_assisted=self.assisted,
                end_reason='task_terminal' if result.terminated else
                'time_limit' if truncated else None)
            self.observation = deepcopy(result.observation)
            self.steps += 1
            if result.terminated or truncated:
                self.active = False
                self.backend.stop()
            return deepcopy(row)
        except Exception as error:
            self.active = False
            try:
                self.backend.stop()
            except Exception as stop_error:
                logging.error('Backend stop also failed: %s', stop_error)
            raise

    def close(self):
        self.active = False
        self.backend.stop()
