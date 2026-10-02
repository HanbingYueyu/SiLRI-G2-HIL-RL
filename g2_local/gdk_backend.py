"""GDK RGB/state reader. Construction is explicit and never enables motion."""
from pathlib import Path
import sys
import time
import threading
import math
import numpy as np
from .contract import vector


class CameraUnavailable(TimeoutError):
    """No qualified camera observation arrived within the bounded reader wait."""


class _SerializedRobot:
    """Serialize calls to one non-thread-safe SDK Robot, shared by reader/sender.

    Lock individual Robot calls, not camera acquisition or PTP IPC. The command
    port also holds this reentrant lock across its final checks and target send,
    so waiting for feedback cannot send a target after watchdog cancellation.
    """
    def __init__(self, robot):
        self._robot = robot
        self._sdk_call_lock = threading.RLock()

    def __getattr__(self, name):
        method = getattr(self._robot, name)
        if not callable(method):
            return method

        def call(*args, **kwargs):
            with self._sdk_call_lock:
                return method(*args, **kwargs)
        return call


def source_timestamp_ns(value):
    if (isinstance(value, bool) or not isinstance(value, (int, np.integer)) or
            value <= 0):
        raise ValueError('Invalid source timestamp: positive integer nanoseconds required')
    return int(value)


def measured_pose(pose):
    values = np.asarray((*pose.position_m, *pose.orientation_xyzw), dtype=float)
    return validate_pose(values)


def validate_pose(values):
    values = np.asarray(values, dtype=float)
    if (values.shape != (7,) or not np.isfinite(values).all() or
            abs(np.linalg.norm(values[3:]) - 1.) > .01):
        raise ValueError('Invalid TF/motion pose')
    return values


def transform_pose(transform):
    values = [float(getattr(transform.translation, k)) for k in 'xyz'] + [
        float(getattr(transform.rotation, k)) for k in 'xyzw']
    return validate_pose(values).tolist()


def query_tf_directions(tf):
    queries = []
    for target, source in (('base_link', 'arm_l_end_link'), ('arm_l_end_link', 'base_link')):
        transform, stamp = tf.lookup_transform_latest(target, source, True)
        queries.append(dict(target=target, source=source, timestamp_ns=source_timestamp_ns(stamp),
                            pose=transform_pose(transform)))
    return queries


def wait_for_tf(tf, *, timeout_s=10., retry_s=.05):
    """Boundedly retry until both directions are available in the same attempt."""
    if not 0 < timeout_s <= 30 or not 0 < retry_s <= timeout_s:
        raise ValueError('Invalid TF preflight timeout')
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            return query_tf_directions(tf)
        except RuntimeError as exc:
            last_error = exc
            time.sleep(min(retry_s, max(0, deadline - time.monotonic())))
    raise TimeoutError('TF cache did not expose both arm_l_end_link directions') from last_error


def tf_motion_evidence(tf, pose):
    """Use the installed SDK's verified direction; never swap to hide a mismatch.

    Motion has no independent source timestamp. Its pose is only cross-checked
    against TF; the freshness guard owns the caller-supplied error thresholds.
    """
    pose = validate_pose(pose)
    queries = query_tf_directions(tf)
    candidate = np.asarray(queries[1]['pose'])
    position_error = math.dist(candidate[:3], pose[:3])
    if not math.isfinite(position_error):
        raise ValueError('Invalid TF/motion pose difference')
    dot = float(abs(np.dot(candidate[3:] / np.linalg.norm(candidate[3:]),
                           pose[3:] / np.linalg.norm(pose[3:]))))
    return dict(tf_queries=queries, tf_position_error_m=position_error,
                tf_rotation_error_rad=2 * math.acos(min(1., dot)))


class GdkCommandPort:
    """Low-level diagnostic command boundary, NOT a complete Gym backend.

    No automatic mode changes, gripper writes or background publication.
    Caller must supply validated targets, verified source freshness checks
    returning exactly True (otherwise raise/reject),
    exclusive ownership and a bounded control loop. The SDK sender is the
    inspected local reference adapter's internal primitive (version-sensitive).

    stop latches first; when healthy it sends ONE measured-pose hold. That is
    not a physical emergency stop. With failed communication no hold is sent;
    firmware expiry behavior and stopping distance require commissioning.
    A blocked SDK call cannot be interrupted by this Python lock.
    """
    def __init__(self, controller, *, expected_mode, allow_motion=False,
                 freshness_guard=None, life_time_s=.1, stop_pose_provider=None,
                 safe_stop_request=None):
        if type(expected_mode) is not int or expected_mode not in (1, 3):
            raise ValueError('Explicit control mode 1 or 3 required')
        if type(allow_motion) is not bool:
            raise ValueError('Explicit boolean motion permission required')
        if allow_motion and not callable(freshness_guard):
            raise ValueError('Verified feedback freshness guard required for motion')
        if not math.isfinite(life_time_s) or not .04 <= life_time_s <= .2:
            raise ValueError('Invalid command lifetime')
        self.controller = controller
        self.expected_mode = expected_mode
        self.enabled = allow_motion
        self.guard = freshness_guard
        if stop_pose_provider is not None and not callable(stop_pose_provider):
            raise ValueError('Stop pose provider must be callable')
        self.stop_pose_provider = stop_pose_provider
        if safe_stop_request is not None and not callable(safe_stop_request):
            raise ValueError('Safe stop request must be callable')
        self.safe_stop_request = safe_stop_request
        self.safe_stop_acknowledged = False
        self.safe_stop_error = None
        self.stop_attempted = False
        self.life_time = life_time_s
        self.stopped = False
        self.attempted = False
        self.stop_failure = None
        self.cancel_event = None
        self.lock = threading.RLock()
        self._sdk_lock = getattr(getattr(controller, 'robot', None),
                                 '_sdk_call_lock', threading.RLock())

    def bind_cancel_event(self, cancel_event):
        if not callable(getattr(cancel_event, 'is_set', None)):
            raise ValueError('A command-stream cancellation event is required')
        with self.lock:
            if self.cancel_event is not None and self.cancel_event is not cancel_event:
                raise RuntimeError('Command port is already owned by another stream')
            self.cancel_event = cancel_event

    def _check_cancelled(self):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise RuntimeError('Command stream halted before SDK target send')

    def _check_health(self):
        self.controller.checked_arm_state()
        status = self.controller.motion_status_summary()
        if status['control_mode'] != self.expected_mode or status['error_code'] != 0:
            raise RuntimeError(f'Unhealthy or changed control mode: {status}')
    def _check(self):
        self._check_health()
        if self.guard() is not True:
            # Carry the freshness decision code so the caller can tell a
            # recoverable camera fault from a liveness fault, and so the log
            # names the actual reason instead of a generic message.
            decision = getattr(getattr(self.guard, '__self__', None), 'last_decision', None)
            code = getattr(decision, 'code', None)
            error = RuntimeError('Feedback freshness not explicitly confirmed'
                                 + (f': {code}' if isinstance(code, str) and code else ''))
            if isinstance(code, str) and code:
                error.code = code
            raise error

    def _write(self, target, *, target_send=False):
        vector(target.position_m, 3)
        q = vector(target.orientation_xyzw, 4)
        if abs(np.linalg.norm(q) - 1.) > .01:
            raise ValueError('Target quaternion must be unit length')
        if target_send:
            self._check_cancelled()
        self.controller._send_left_cartesian_pose(target, self.life_time)

    def send(self, target):
        with self.lock, self._sdk_lock:
            if not self.enabled:
                raise PermissionError('Motion disabled; no GDK command sent')
            if self.stopped:
                raise RuntimeError('Command port stopped; explicit reconstruction required')
            try:
                self._check()
                self._check_cancelled()
                self.attempted = True
                self._write(target, target_send=True)
            except Exception as error:
                self.stopped = True
                self.stop_failure = error
                raise

    def stop(self):
        with self.lock:
            if self.stop_attempted:
                if self.stop_failure is not None:
                    raise RuntimeError('physical stop unconfirmed after failed send') from self.stop_failure
                return
            self.stop_attempted = True
            self.stopped = True
            if not self.enabled or not self.attempted:
                if self.stop_failure is not None:
                    raise RuntimeError('physical stop unconfirmed after failed send') from self.stop_failure
                return
            try:
                if self.stop_pose_provider is None:
                    self._check()
                    measured = self.controller.read_end_effector_pose('arm_l_end_link')
                    self._check()
                else:
                    self._check_health()
                    measured = self.stop_pose_provider()
                    self._check_health()
                self._write(measured)
                self.stop_failure = None
            except Exception as error:
                self.stop_failure = error
                # This request does not depend on the failed pose/PTP guard.
                # Never resume targets or claim physical stop from an SDK ACK.
                if self.safe_stop_request is not None:
                    try:
                        result = self.safe_stop_request()
                        if type(result) is not int or result != 0:
                            raise RuntimeError('Invalid SAFE_STOP acknowledgement')
                        self.safe_stop_acknowledged = True
                    except Exception as stop_error:
                        self.safe_stop_error = stop_error
                    detail = ('left SAFE_STOP acknowledged; physical stop unconfirmed'
                              if self.safe_stop_acknowledged else
                              f'left SAFE_STOP failed: {self.safe_stop_error}; physical stop unconfirmed')
                    raise RuntimeError(detail) from error
                raise RuntimeError('physical stop unconfirmed: measured hold failed') from error


class GdkReader:
    """Read-only observations with transactionally committed evidence.

    last_info describes only the last *successful* observe(), or is empty
    before the first success. A raised read never updates it or last_stamps;
    callers must not treat retained evidence as a new observation.
    """
    def __init__(self, adapter_root='/home/flyfuture/g2_hinge_assembly', timeout_s=2.,
                 *, allow_motion=False):
        if type(allow_motion) is not bool:
            raise ValueError('Explicit boolean GDK motion intent required')
        if not 0 < timeout_s <= 30:
            raise ValueError('Invalid reader timeout: must be in (0, 30] seconds')
        root = Path(adapter_root).resolve()
        if not (root / 'g2_adapter/control.py').is_file():
            raise FileNotFoundError(f'Missing validated G2 adapter: {root}')
        sys.path.insert(0, str(root))
        from g2_adapter.control import G2Controller
        from g2_adapter.camera import decode_color_rgb
        import agibot_gdk as gdk
        self.gdk = gdk
        self.decode = decode_color_rgb
        self.timeout_s = timeout_s
        self.camera_pair_max_skew_s = None
        self.closed = True
        self.last_stamps = {}
        self.last_info = {}
        # Upstream-style freshness: the local monotonic time at which each
        # source's robot timestamp last *changed*.  A repeated, frozen or
        # cached frame keeps its previous value, so its age keeps growing
        # without any robot-clock-to-local-clock mapping.
        self._source_stamp = {}
        self._source_changed_mono = {}
        try:
            # Own the initialization attempt: even an interrupted/failed SDK
            # init may have allocated resources before control returns here.
            self.closed = False
            if gdk.gdk_init() != gdk.GDKRes.kSuccess:
                raise RuntimeError('GDK initialization failed')
            self.robot = _SerializedRobot(gdk.Robot())
            self.controller = G2Controller(gdk, self.robot, allow_motion=allow_motion)
            self.streams = {'left_wrist': gdk.CameraType.kHandLeftColor,
                            'right_aux': gdk.CameraType.kHandRightColor}
            self.camera = gdk.Camera(list(self.streams.values()))
            self.tf = gdk.TF()
            wait_for_tf(self.tf, timeout_s=timeout_s, retry_s=min(.005, timeout_s))
            time.sleep(1)
        except BaseException:
            try:
                self.close()
            except BaseException as error:
                # Never let a cleanup interrupt masquerade as a clean Ctrl+C
                # in the caller, which has not yet received this reader.
                raise RuntimeError(f'GDK initialization cleanup failed: {type(error).__name__}: {error}') from error
            raise

    def read_control_pose(self):
        """Read-only drift feedback; does not wait for or consume camera frames.

        This is not an observation/freshness approval. The caller must still
        revalidate the timestamped predecessor before submitting any target.
        """
        if self.closed:
            raise RuntimeError('Reader is closed')
        self.controller.checked_arm_state()
        return measured_pose(self.controller.read_end_effector_pose('arm_l_end_link'))

    def observe_with_timeout(self, timeout_s):
        return self.observe(timeout_s=timeout_s)

    def read_stop_feedback(self):
        """Independent measured hold evidence; no camera call, no commands."""
        start = time.monotonic_ns()
        wall_start = time.time_ns()
        pose = self.read_control_pose()
        evidence = tf_motion_evidence(self.tf, pose)
        evidence.update(joint_timestamp_ns=source_timestamp_ns(self.robot.get_joint_states()['timestamp']),
                        read_start_monotonic_ns=start, read_start_wall_ns=wall_start,
                        motion_pose=pose.tolist())
        return evidence

    def observe(self, *, timeout_s=None):
        if self.closed:
            raise RuntimeError('Reader is closed')
        timeout_s = self.timeout_s if timeout_s is None else timeout_s
        if (type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or
                not 0 < timeout_s <= self.timeout_s):
            raise ValueError('Invalid camera observation timeout')
        start_mono = time.monotonic_ns()
        start_wall = time.time_ns()
        start_sdk = source_timestamp_ns(self.gdk.Clock.now_ns())
        obs = {}
        stamps = {}
        frames = {}
        deadline = time.monotonic() + timeout_s
        camera_calls = {key: 0 for key in self.streams}
        camera_wait_s = {key: 0. for key in self.streams}
        camera_max_call_s = {key: 0. for key in self.streams}
        repeats = {key: 0 for key in self.streams}
        last_camera_error = None
        while True:
            # Refresh BOTH candidates while waiting, rather than pinning the
            # left frame while the right stream catches up. Decode only the
            # final pair; original capture timestamps remain unchanged.
            for key, stream in self.streams.items():
                call_start = time.monotonic()
                remaining_s = deadline-call_start
                if remaining_s <= 0:
                    if len(stamps) == len(self.streams):
                        skew_ns = max(stamps.values())-min(stamps.values())
                        if (self.camera_pair_max_skew_s is not None and
                                skew_ns > self.camera_pair_max_skew_s*1e9):
                            raise CameraUnavailable(
                                f'Camera pair unavailable: skew_ms={skew_ns/1e6:.3f}; '
                                f'limit_ms={self.camera_pair_max_skew_s*1000:.3f}; '
                                f'camera_calls={camera_calls}; repeated_frames={repeats}; '
                                f'sdk_total_s={camera_wait_s}; '
                                f'sdk_max_call_s={camera_max_call_s}; '
                                f'capture_timestamps_ns={stamps}')
                    message = (f'Camera acquisition exceeded its bounded wait: '
                               f'camera_calls={camera_calls}; repeated_frames={repeats}')
                    if last_camera_error is not None:
                        raise CameraUnavailable(message) from last_camera_error
                    raise CameraUnavailable(message)
                try:
                    remaining_ms = max(.1, remaining_s*1000.)
                    frame = self.camera.get_latest_image(stream, min(100., remaining_ms))
                except Exception as error:
                    elapsed = time.monotonic()-call_start
                    last_camera_error = error
                    camera_calls[key] += 1
                    camera_wait_s[key] += elapsed
                    camera_max_call_s[key] = max(camera_max_call_s[key], elapsed)
                    if time.monotonic() >= deadline:
                        raise CameraUnavailable(
                            f'Camera image acquisition failed before deadline: camera={key}; '
                            f'calls={camera_calls}; repeated_frames={repeats}') from error
                    time.sleep(.005)
                    break
                elapsed = time.monotonic()-call_start
                camera_calls[key] += 1
                camera_wait_s[key] += elapsed
                camera_max_call_s[key] = max(camera_max_call_s[key], elapsed)
                stamp = source_timestamp_ns(frame.timestamp_ns)
                repeats[key] += int(stamp == stamps.get(key))
                frames[key], stamps[key] = frame, stamp
            else:
                last_camera_error = None
            if last_camera_error is not None:
                if time.monotonic() >= deadline:
                    raise CameraUnavailable(
                        f'Camera image acquisition failed before deadline: '
                        f'camera_calls={camera_calls}; repeated_frames={repeats}') from last_camera_error
                continue
            skew_ns = max(stamps.values()) - min(stamps.values())
            mismatched = (self.camera_pair_max_skew_s is not None and
                          skew_ns > self.camera_pair_max_skew_s * 1e9)
            if not mismatched:
                break
            if mismatched:
                if time.monotonic() >= deadline:
                    raise CameraUnavailable(
                        f'Camera pair unavailable: skew_ms={skew_ns/1e6:.3f}; '
                        f'limit_ms={self.camera_pair_max_skew_s*1000:.3f}; '
                        f'camera_calls={camera_calls}; repeated_frames={repeats}; '
                        f'sdk_total_s={camera_wait_s}; sdk_max_call_s={camera_max_call_s}; '
                        f'capture_timestamps_ns={stamps}')
            time.sleep(.005)
        acquired_mono = time.monotonic_ns()
        for key, frame in frames.items():
            obs[key] = self.decode(frame, self.gdk)
        decoded_mono = time.monotonic_ns()
        self.controller.checked_arm_state()
        pose = measured_pose(self.controller.read_end_effector_pose('arm_l_end_link'))
        state_received = time.monotonic_ns()
        if np.any(np.abs(pose) > np.finfo(np.float32).max):
            raise ValueError('Invalid measured pose for float32 observation')
        obs['state'] = pose.astype(np.float32)
        sources = dict(stamps, joint=source_timestamp_ns(self.robot.get_joint_states()['timestamp']))
        evidence = tf_motion_evidence(self.tf, pose)
        sources['tf'] = evidence['tf_queries'][1]['timestamp_ns']
        observed_mono = time.monotonic_ns()
        for key, stamp in sources.items():
            if self._source_stamp.get(key) != stamp:
                self._source_stamp[key] = stamp
                self._source_changed_mono[key] = observed_mono
        status = self.controller.motion_status_summary()
        end_sdk = source_timestamp_ns(self.gdk.Clock.now_ns())
        end_wall = time.time_ns()
        end_mono = time.monotonic_ns()
        info = dict(evidence, camera_timestamp_ns=dict(stamps), source_timestamp_ns=sources,
                    source_changed_monotonic_ns=dict(self._source_changed_mono),
                    read_start_monotonic_ns=start_mono, read_start_wall_ns=start_wall,
                    read_start_sdk_clock_ns=start_sdk,
                    read_end_monotonic_ns=end_mono, read_end_wall_ns=end_wall,
                    read_end_sdk_clock_ns=end_sdk, sdk_clock_ns=end_sdk,
                    motion_pose=pose.tolist(), state_received_monotonic_ns=state_received,
                    received_monotonic_ns=end_mono, read_duration_s=(end_mono - start_mono) / 1e9,
                    camera_acquire_s=(acquired_mono-start_mono)/1e9,
                    camera_decode_s=(decoded_mono-acquired_mono)/1e9,
                    camera_calls=camera_calls, camera_repeated_frames=repeats,
                    camera_reused_frames={
                        key: int(stamps[key] <= self.last_stamps.get(key, 0))
                        for key in self.streams},
                    camera_sdk_total_s=camera_wait_s, camera_sdk_max_call_s=camera_max_call_s,
                    motion_status=status, backend='gdk_read_only')
        self.last_stamps = stamps
        self.last_info = info
        return obs

    def read_sdk_clock_ns(self):
        """Robot/SDK clock reading, the same domain as the sensor timestamps.

        Used to anchor a command send so a successor observation can be checked
        with ``successor_timestamp_ns > send_sdk_clock_ns`` inside ONE clock
        domain (no robot-to-local mapping, no PTP). The freshness guard decides
        whether the reading is actually comparable with the observation stamps.
        """
        if self.closed:
            raise RuntimeError('Reader is closed')
        return source_timestamp_ns(self.gdk.Clock.now_ns())

    def close(self):
        if not self.closed:
            self.closed = True
            self.camera = None
            self.tf = None
            self.controller = None
            self.robot = None
            self.gdk.gdk_release()
