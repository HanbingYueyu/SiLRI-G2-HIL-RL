"""GDK RGB/state reader. Construction is explicit and never enables motion."""
from pathlib import Path
import sys
import time
import threading
import math
import numpy as np
from .contract import vector


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
                 freshness_guard=None, life_time_s=.1):
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
        self.life_time = life_time_s
        self.stopped = False
        self.attempted = False
        self.lock = threading.RLock()

    def _check(self):
        self.controller.checked_arm_state()
        status = self.controller.motion_status_summary()
        if status['control_mode'] != self.expected_mode or status['error_code'] != 0:
            raise RuntimeError(f'Unhealthy or changed control mode: {status}')
        if self.guard() is not True:
            raise RuntimeError('Feedback freshness not explicitly confirmed')

    def _write(self, target):
        vector(target.position_m, 3)
        q = vector(target.orientation_xyzw, 4)
        if abs(np.linalg.norm(q) - 1.) > .01:
            raise ValueError('Target quaternion must be unit length')
        self.controller._send_left_cartesian_pose(target, self.life_time)

    def send(self, target):
        with self.lock:
            if not self.enabled:
                raise PermissionError('Motion disabled; no GDK command sent')
            if self.stopped:
                raise RuntimeError('Command port stopped; explicit reconstruction required')
            try:
                self._check()
                self.attempted = True
                self._write(target)
            except Exception:
                self.stopped = True
                raise

    def stop(self):
        with self.lock:
            if self.stopped:
                return
            self.stopped = True
            if not self.enabled or not self.attempted:
                return
            self._check()
            measured = self.controller.read_end_effector_pose('arm_l_end_link')
            self._check()
            self._write(measured)


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
        self.closed = True
        self.last_stamps = {}
        self.last_info = {}
        try:
            # Own the initialization attempt: even an interrupted/failed SDK
            # init may have allocated resources before control returns here.
            self.closed = False
            if gdk.gdk_init() != gdk.GDKRes.kSuccess:
                raise RuntimeError('GDK initialization failed')
            self.robot = gdk.Robot()
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

    def observe(self):
        if self.closed:
            raise RuntimeError('Reader is closed')
        start_mono = time.monotonic_ns()
        start_wall = time.time_ns()
        start_sdk = source_timestamp_ns(self.gdk.Clock.now_ns())
        self.controller.checked_arm_state()
        pose = measured_pose(self.controller.read_end_effector_pose('arm_l_end_link'))
        state_received = time.monotonic_ns()
        if np.any(np.abs(pose) > np.finfo(np.float32).max):
            raise ValueError('Invalid measured pose for float32 observation')
        state = pose.astype(np.float32)
        obs = {'state': state}
        stamps = {}
        for key, stream in self.streams.items():
            deadline = time.monotonic() + self.timeout_s
            while True:
                frame = self.camera.get_latest_image(stream, 100.)
                stamp = source_timestamp_ns(frame.timestamp_ns)
                if stamp > self.last_stamps.get(key, 0):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'No advancing camera timestamp: {key}')
                time.sleep(.005)
            obs[key] = self.decode(frame, self.gdk)
            stamps[key] = stamp
        sources = dict(stamps, joint=source_timestamp_ns(self.robot.get_joint_states()['timestamp']))
        evidence = tf_motion_evidence(self.tf, pose)
        sources['tf'] = evidence['tf_queries'][1]['timestamp_ns']
        status = self.controller.motion_status_summary()
        end_sdk = source_timestamp_ns(self.gdk.Clock.now_ns())
        end_wall = time.time_ns()
        end_mono = time.monotonic_ns()
        info = dict(evidence, camera_timestamp_ns=dict(stamps), source_timestamp_ns=sources,
                    read_start_monotonic_ns=start_mono, read_start_wall_ns=start_wall,
                    read_start_sdk_clock_ns=start_sdk,
                    read_end_monotonic_ns=end_mono, read_end_wall_ns=end_wall,
                    read_end_sdk_clock_ns=end_sdk, sdk_clock_ns=end_sdk,
                    motion_pose=pose.tolist(), state_received_monotonic_ns=state_received,
                    received_monotonic_ns=end_mono, read_duration_s=(end_mono - start_mono) / 1e9,
                    motion_status=status, backend='gdk_read_only')
        self.last_stamps = stamps
        self.last_info = info
        return obs

    def close(self):
        if not self.closed:
            self.closed = True
            self.camera = None
            self.tf = None
            self.controller = None
            self.robot = None
            self.gdk.gdk_release()
