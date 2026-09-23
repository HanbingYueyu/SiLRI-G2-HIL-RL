"""Serialized reset, step, and terminal decisions for the real actor."""

from dataclasses import dataclass
import math
import time
import uuid
import numpy as np

from .contract import CAMERA_KEYS, EpisodeContext, vector
from .operator_control import StartChord, TerminalKeyReader
from .outcome import OutcomeDecision


@dataclass(frozen=True)
class StepToken:
    episode_id: str
    step_id: int
    nonce: str


class RealEpisodeCoordinator:
    def __init__(self, intervention, keys, task, *, context_max_age_s,
                 clock_ns=time.monotonic_ns, left_button=0, right_button=1):
        if not math.isfinite(context_max_age_s) or context_max_age_s <= 0:
            raise ValueError('Positive finite context maximum age required')
        self.intervention = intervention
        self.keys = keys if isinstance(keys, TerminalKeyReader) else TerminalKeyReader(keys)
        self.task = task
        self.context_max_age_s = context_max_age_s
        self.clock_ns = clock_ns
        self.chord = StartChord(left_button=left_button, right_button=right_button)
        self.state = 'WAITING_FOR_RESET'
        self.context = None
        self._token = None
        self._completed_token = None
        self._pending_terminal = None
        self._step_id = 0
        self.success_reward = task.success_reward
        self.failure_reward = task.failure_reward
        self.step_reward = task.step_reward

    @property
    def running(self):
        return self.state == 'RUNNING'

    @property
    def completed_step_token(self):
        return self._completed_token

    @property
    def active_step_token(self):
        return self._token

    def _validate_context(self, context):
        if not isinstance(context, EpisodeContext):
            raise TypeError('EpisodeContext required')
        stamp = context.visual_reset_monotonic_ns
        if stamp is not None:
            age = self.clock_ns() - stamp
            if age < 0 or age > self.context_max_age_s * 1_000_000_000:
                raise ValueError('stale reset context')
        if (any(abs(value) > self.task.target_xy_range_m
                for value in context.target_offset_m[:2]) or
                context.target_offset_m[2] != 0):
            raise ValueError('target offset outside configured range')
        if (any(abs(value) > self.task.ee_xyz_range_m
                for value in context.ee_reset_offset[:3]) or
                any(abs(value) > self.task.ee_rpy_range_rad
                    for value in context.ee_reset_offset[3:])):
            raise ValueError('EE reset offset outside configured range')

    def offer_context(self, context):
        if self.state != 'WAITING_FOR_RESET':
            raise RuntimeError('Episode is not waiting for reset')
        self._validate_context(context)
        self.context = context
        self._completed_token = None
        self.chord.reset()

    def observe_start_frame(self):
        if self.state != 'WAITING_FOR_RESET' or self.context is None:
            return False
        if getattr(self.intervention, 'fault', None) is not None or self.intervention.last_frame is None:
            self._abort()
            raise RuntimeError('Input fault before episode start')
        gate = getattr(self.intervention, 'gate', None)
        if gate is None or getattr(gate, 'fresh', None) is not True:
            self.chord.reset()
            return False
        if not self.chord.update(self.intervention.last_frame):
            return False
        self._validate_context(self.context)
        if self.context.visual_reset_monotonic_ns is None:
            raise ValueError('stale reset context: missing visual reset time')
        self.keys.drain()
        self._step_id = 0
        self._pending_terminal = None
        self.chord.reset()
        self.state = 'RUNNING'
        return True

    def begin_step(self):
        if not self.running:
            raise RuntimeError('Episode not running')
        if self._token is not None:
            raise RuntimeError('Step already active')
        if getattr(self.intervention, 'fault', None) is not None or self.intervention.last_frame is None:
            self._abort()
            raise RuntimeError('Input fault during episode')
        self._token = StepToken(self.context.episode_id, self._step_id, uuid.uuid4().hex)
        self._completed_token = None
        return self._token

    def request_terminal(self, label):
        if not self.running:
            raise RuntimeError('Episode not running')
        if label not in ('success', 'failure'):
            raise ValueError('Unknown terminal label')
        if self._pending_terminal is not None and self._pending_terminal != label:
            self._abort()
            raise RuntimeError('Conflicting Y/F terminal input')
        self._pending_terminal = label

    def _validate_token(self, token):
        if not self.running or self._token is None or token != self._token:
            raise RuntimeError('Invalid or inactive step token')

    def _finish(self, decision):
        self._completed_token = self._token
        self._token = None
        self._pending_terminal = None
        self._step_id += 1
        if decision.terminated:
            self.state = 'WAITING_FOR_RESET'
            self.context = None
            self.chord.reset()
        return decision

    def seal_episode(self, token):
        """Seal a Gym time-limit only after its valid successor completed."""
        if self._token is not None:
            raise RuntimeError('Cannot seal episode with active step')
        if (not self.running or self._completed_token is None or
                token != self._completed_token):
            raise RuntimeError('Invalid completed step token')
        self.state = 'WAITING_FOR_RESET'
        self.context = None
        self._completed_token = None
        self.chord.reset()

    def _finish_step(self, token):
        self._validate_token(token)
        if self._pending_terminal == 'success':
            return self._finish(OutcomeDecision(self.success_reward, True, 'human', True))
        if self._pending_terminal == 'failure':
            return self._finish(OutcomeDecision(self.failure_reward, True, 'human', False))
        return self._finish(OutcomeDecision(self.step_reward, False, 'human', None))

    @staticmethod
    def _validate_successor(observation):
        if not isinstance(observation, dict) or set(observation) != {'state', *CAMERA_KEYS}:
            raise ValueError('Valid G2 successor observation required')
        try:
            pose = vector(observation['state'], 7)
        except (TypeError, ValueError) as exc:
            raise ValueError('Invalid successor state') from exc
        if abs(np.linalg.norm(pose[3:]) - 1.) > .01:
            raise ValueError('Invalid successor quaternion')
        for key in CAMERA_KEYS:
            image = observation[key]
            if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
                    image.ndim != 3 or image.shape[2] != 3 or
                    min(image.shape[:2]) <= 0):
                raise ValueError('Invalid successor RGB observation')

    def outcome(self, observation):
        self._validate_successor(observation)
        token = self._token
        self._validate_token(token)
        if getattr(self.intervention, 'fault', None) is not None or self.intervention.last_frame is None:
            self._abort()
            raise RuntimeError('Input fault during episode')
        try:
            label = self.keys.poll()
            if label is not None:
                self.request_terminal(label)
            return self._finish_step(token)
        except Exception:
            self._abort()
            raise

    def _abort(self):
        self.state = 'ABORTED'
        self.context = None
        self._token = None
        self._completed_token = None
        self._pending_terminal = None
        self.chord.reset()

    def abort_step(self, token):
        self._validate_token(token)
        self._abort()
