"""Gym execution path over reader + GdkCommandPort, disabled by default.

No constructor opens hardware, enables a controller, or chooses a control mode.
Guards must verify source acquisition timestamps, not just local receive time.
"""
from copy import deepcopy
from dataclasses import dataclass
import logging
import math
import threading
import time
import numpy as np
from .command_stream import CommandStream
from .contract import CAMERA_KEYS, vector
from .episode import StepResult
from .motion import plan_target
from .outcome import coerce_outcome


@dataclass(frozen=True)
class PoseTarget:
    position_m: tuple
    orientation_xyzw: tuple


class MotionBackend:
    name = 'gdk_motion'

    def __init__(self, reader, port, *, config, observation_guard, outcome,
                 command_timeout, send_timeout, step_period, allow_motion=False,
                 stop_timeout=1.):
        config.validate_motion()
        if type(allow_motion) is not bool:
            raise ValueError('Explicit boolean motion permission required')
        if not callable(observation_guard) or not callable(outcome):
            raise ValueError('Observation guard and task outcome callback required')
        if not math.isfinite(step_period) or not 0 < step_period < command_timeout:
            raise ValueError('Step period must be positive and below command lease')
        self.reader, self.config = reader, config
        self.observation_guard, self.outcome = observation_guard, outcome
        self.enabled = allow_motion
        self.step_period = step_period
        self.stream = CommandStream(port, command_timeout=command_timeout,
                                    send_timeout=send_timeout, stop_timeout=stop_timeout)
        self.stopped = self.closed = False
        self.execute_lock = threading.Lock()
        self.reader_lock = threading.RLock()

    def _read(self, *, after=None):
        with self.reader_lock:
            return self._read_locked(after=after)

    def _read_locked(self, *, after=None):
        if self.stopped or self.closed:
            raise RuntimeError('Motion backend stopped; reconstruct explicitly before rearming')
        self.stream.check()
        obs = self.reader.observe()
        info = deepcopy(getattr(self.reader, 'last_info', {}))
        if set(obs) != {'state', *CAMERA_KEYS}:
            raise ValueError('Missing or unexpected observation fields')
        pose = vector(obs['state'], 7)
        if abs(np.linalg.norm(pose[3:])-1.) > .01:
            raise ValueError('Invalid observed quaternion')
        for key in CAMERA_KEYS:
            array = obs[key]
            if (array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3
                    or min(array.shape[:2]) <= 0):
                raise ValueError('Invalid RGB observation')
        if self.observation_guard(obs, info, after) is not True:
            message = 'Source observation freshness not confirmed'
            try:
                code = self.observation_guard.last_decision.code
            except Exception:
                code = None
            if (type(code) is str and 0 < len(code) <= 128 and code.isascii() and
                    all(character.isalnum() or character in '_:-.' for character in code)):
                message += f': {code}'
            raise RuntimeError(message)
        self.stream.check()  # Camera may have blocked past the target lease.
        return deepcopy(obs)

    def _abort(self):
        try:
            self.stop()
        except Exception:
            logging.exception('Stop unconfirmed while handling execution failure')

    def observe(self):
        try:
            return self._read()
        except Exception:
            self._abort()
            raise

    def execute(self, action):
        if not self.enabled:
            raise PermissionError('Motion disabled; no command submitted')
        if not self.execute_lock.acquire(blocking=False):
            raise RuntimeError('Concurrent execute is not supported')
        try:
            before = self._read()
            pose, effective = plan_target(before['state'], action, self.config)
            sequence = self.stream.submit(PoseTarget(pose[:3], pose[3:]))
            sent_at = self.stream.wait_sent(sequence, timeout=self.stream.command_timeout)
            self.stream.halt.wait(max(0., sent_at+self.step_period-time.monotonic()))
            self.stream.check()
            after = self._read(after=sent_at)
            decision = coerce_outcome(self.outcome(after))
            self.stream.check()
            return StepResult(after, effective, decision.reward, decision.terminated,
                              decision.reward_source, decision.success_label)
        except Exception:
            self._abort()
            raise
        finally:
            self.execute_lock.release()

    def stop(self):
        self.stopped = True
        self.stream.stop()

    def close(self):
        if self.closed:
            return
        self.stop()  # On failure, retain SDK resources while a send/hold may be alive.
        if not self.reader_lock.acquire(timeout=self.stream.stop_timeout):
            raise TimeoutError('Source observation still in flight; SDK resources retained')
        try:
            if not self.closed:
                self.reader.close()
                self.closed = True
        finally:
            self.reader_lock.release()
