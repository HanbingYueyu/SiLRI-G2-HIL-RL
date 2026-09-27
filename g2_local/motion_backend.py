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
from scipy.spatial.transform import Rotation
from .command_stream import CommandStream
from .contract import CAMERA_KEYS, vector
from .episode import StepResult
from .motion import plan_target
from .outcome import coerce_outcome


@dataclass(frozen=True)
class PoseTarget:
    position_m: tuple
    orientation_xyzw: tuple


class ObservationRejected(RuntimeError):
    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


class MotionBackend:
    name = 'gdk_motion'

    def __init__(self, reader, port, *, config, observation_guard, outcome,
                 command_timeout, send_timeout, step_period, allow_motion=False,
                 stop_timeout=1., send_rate_hz=50., local_envelope=None,
                 reference_guard=None, before_command=None,
                 policy_position_drift_m=.005, policy_rotation_drift_rad=.02,
                 startup_camera_wait_s=0., startup_mapping_wait_s=0.):
        config.validate_motion()
        if type(allow_motion) is not bool:
            raise ValueError('Explicit boolean motion permission required')
        if not callable(observation_guard) or not callable(outcome):
            raise ValueError('Observation guard and task outcome callback required')
        if not math.isfinite(step_period) or not 0 < step_period < command_timeout:
            raise ValueError('Step period must be positive and below command lease')
        self.reader, self.config = reader, config
        if not math.isfinite(startup_camera_wait_s) or not 0 <= startup_camera_wait_s <= 10:
            raise ValueError('Startup camera wait must be in [0,10] seconds')
        self.startup_camera_wait_s = startup_camera_wait_s
        if not math.isfinite(startup_mapping_wait_s) or not 0 <= startup_mapping_wait_s <= 120:
            raise ValueError('Startup mapping wait must be in [0,120] seconds')
        self.startup_mapping_wait_s = startup_mapping_wait_s
        for value in (policy_position_drift_m, policy_rotation_drift_rad):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('Policy pose drift limits must be positive and finite')
        self.policy_position_drift_m = policy_position_drift_m
        self.policy_rotation_drift_rad = policy_rotation_drift_rad
        self.local_envelope = local_envelope
        self.episode_reference = None
        self.reference_guard = reference_guard
        self.before_command = before_command
        self._last_accepted = None
        self.last_execution_timing = {}
        self.observation_guard, self.outcome = observation_guard, outcome
        self.enabled = allow_motion
        self.step_period = step_period
        self.stream = CommandStream(port, command_timeout=command_timeout,
                                    send_timeout=send_timeout, stop_timeout=stop_timeout,
                                    rate_hz=send_rate_hz)
        self.stopped = self.closed = False
        self.execute_lock = threading.Lock()
        self.reader_lock = threading.RLock()

    def _read(self, *, after=None):
        with self.reader_lock:
            return self._read_locked(after=after)

    def _check_local(self, pose, stage, *, reference=None):
        reference = self.episode_reference if reference is None else reference
        if self.local_envelope is None or reference is None:
            return
        try:
            self.local_envelope.check(pose, reference)
        except ValueError as error:
            error.boundary_event = dict(
                stage=stage, reason=str(error), pose=list(pose),
                episode_reference=list(reference),
                translation_low_m=list(self.local_envelope.translation_low_m),
                translation_high_m=list(self.local_envelope.translation_high_m),
                rotation_max_rad=self.local_envelope.rotation_max_rad)
            raise

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
                owner = getattr(self.observation_guard, '__self__', self.observation_guard)
                code = owner.last_decision.code
            except Exception:
                code = None
            if (type(code) is str and 0 < len(code) <= 128 and code.isascii() and
                    all(character.isalnum() or character in '_:-.' for character in code)):
                message += f': {code}'
            if code in ('camera_stale:left_wrist', 'camera_stale:right_aux'):
                decision = owner.last_decision
                ages = dict(getattr(decision, 'age_intervals_s', {}))
                message += (f'; age_intervals_s={ages}; '
                            f'read_s={info.get("read_duration_s")}; '
                            f'camera_acquire_s={info.get("camera_acquire_s")}; '
                            f'camera_decode_s={info.get("camera_decode_s")}')
            raise ObservationRejected(message, code)
        self.stream.check()  # Camera may have blocked past the target lease.
        if self.local_envelope is not None and self.episode_reference is not None:
            self._check_local(pose, 'successor_read' if after is not None else 'observation_read')
        self._last_accepted = (deepcopy(obs), info)
        return deepcopy(obs)

    def begin_episode(self, observation):
        """Latch the reset observation once; never move or re-anchor a live episode."""
        if self.local_envelope is None:
            return
        try:
            if self.episode_reference is not None or self.stopped or self.closed:
                raise RuntimeError('Reconstruct backend before starting another episode')
            pose = vector(observation['state'], 7)
            # Zero-action planning also validates the fixed absolute workspace.
            plan_target(pose, (0.,)*6, self.config)
            self._check_local(pose, 'episode_reset', reference=pose)
            self.episode_reference = pose
        except Exception:
            self._abort()
            raise

    def _abort(self):
        try:
            self.stop()
        except Exception:
            logging.exception('Stop unconfirmed while handling execution failure')

    def observe(self):
        try:
            mapping_deadline = time.monotonic() + self.startup_mapping_wait_s
            camera_deadline = None
            camera_attempts = 0
            announced = False
            while True:
                try:
                    return self._read()
                except ObservationRejected as error:
                    if self._last_accepted is not None or self.stream.sequence != 0:
                        raise
                    if error.code in ('mapping_expired', 'mapping_warming_up'):
                        deadline, interval = mapping_deadline, .1
                    elif error.code in ('camera_stale:left_wrist', 'camera_stale:right_aux'):
                        if camera_deadline is None:
                            camera_deadline = time.monotonic()+self.startup_camera_wait_s
                        camera_attempts += 1
                        if camera_attempts >= 5:
                            raise
                        deadline, interval = camera_deadline, .02
                    else:
                        raise
                    if time.monotonic() >= deadline:
                        raise
                    if not announced:
                        print(f'启动观测尚未就绪（{error.code}），有限等待恢复；尚未发送运动命令。', flush=True)
                        announced = True
                    time.sleep(min(interval, max(0., deadline-time.monotonic())))
        except Exception:
            self._abort()
            raise

    def execute_from(self, action, predecessor):
        return self.execute(action, predecessor=predecessor)

    def execute(self, action, *, predecessor=None):
        if not self.enabled:
            raise PermissionError('Motion disabled; no command submitted')
        if not self.execute_lock.acquire(blocking=False):
            raise RuntimeError('Concurrent execute is not supported')
        try:
            if self.local_envelope is not None and self.episode_reference is None:
                raise RuntimeError('Episode reference required before motion')
            reference = None
            if predecessor is not None:
                reference = self._last_accepted
                if reference is None or not np.array_equal(
                        predecessor['state'], reference[0]['state']):
                    raise ValueError('Execution predecessor does not match accepted policy input')
            read_pose = getattr(self.reader, 'read_control_pose', None)
            if reference is not None and self.reference_guard is not None and callable(read_pose):
                # The policy predecessor already contains both camera frames.
                # Waiting for another pair here needlessly ages that input.
                # Read only drift feedback; revalidate the ORIGINAL timestamped
                # observation below, never relabel it with a newer timestamp.
                with self.reader_lock:
                    if self.stopped or self.closed:
                        raise RuntimeError('Motion backend stopped')
                    self.stream.check()
                    pose_feedback = vector(read_pose(), 7)
                    if abs(np.linalg.norm(pose_feedback[3:])-1.) > .01:
                        raise ValueError('Invalid observed quaternion')
                    self.stream.check()
                before = {'state': pose_feedback}
                if self.local_envelope is not None and self.episode_reference is not None:
                    self._check_local(pose_feedback, 'pre_command_feedback')
            else:
                before = self._read()
            # New feedback is still checked, but does not silently change action origin.
            plan_target(before['state'], (0.,)*6, self.config)
            if reference is not None and self.reference_guard is not None:
                if self.reference_guard(*reference) is not True:
                    owner = getattr(self.reference_guard, '__self__', None)
                    decision = getattr(owner, 'last_decision', None)
                    code = getattr(decision, 'code', 'unknown')
                    raise RuntimeError('Policy input expired before command; action discarded: '
                                       + str(code)[:128])
            origin = before['state'] if reference is None else reference[0]['state']
            position_drift = float(np.linalg.norm(np.asarray(before['state'][:3])-origin[:3]))
            rotation_drift = float((Rotation.from_quat(before['state'][3:]) *
                                    Rotation.from_quat(origin[3:]).inv()).magnitude())
            if (position_drift > self.policy_position_drift_m or
                    rotation_drift > self.policy_rotation_drift_rad):
                raise RuntimeError(f'Policy pose drift exceeded: {position_drift:.6f} m, '
                                   f'{rotation_drift:.6f} rad; re-observe before rearming')
            pose, effective = plan_target(origin, action, self.config)
            if self.local_envelope is not None:
                self._check_local(pose, 'pre_command_target')
            if self.before_command is not None:
                self.before_command()
            sequence = self.stream.submit(PoseTarget(pose[:3], pose[3:]))
            sent_at = self.stream.wait_sent(sequence, timeout=self.stream.command_timeout)
            self.stream.halt.wait(max(0., sent_at+self.step_period-time.monotonic()))
            self.stream.check()
            after = self._read(after=sent_at)
            self.last_execution_timing = dict(
                action_mapping=dict(
                    origin_pose=list(origin), selected_action=list(action),
                    executed_action=list(effective),
                    workspace_low=list(self.config.workspace_low),
                    workspace_high=list(self.config.workspace_high),
                    action_scale=list(self.config.action_scale),
                    workspace_clip_m=np.maximum(
                        np.maximum(np.asarray(self.config.workspace_low) -
                                   (np.asarray(origin[:3]) + np.clip(action, -1., 1.)[:3] *
                                    np.asarray(self.config.action_scale[:3])),
                                   (np.asarray(origin[:3]) + np.clip(action, -1., 1.)[:3] *
                                    np.asarray(self.config.action_scale[:3])) -
                                   np.asarray(self.config.workspace_high)), 0.).tolist()),
                policy_position_drift_m=position_drift,
                policy_rotation_drift_rad=rotation_drift,
                command_sent_monotonic_ns=round(sent_at*1e9),
                successor_received_monotonic_ns=time.monotonic_ns())
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
