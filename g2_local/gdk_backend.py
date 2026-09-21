"""GDK RGB/state reader. Construction is explicit and never enables motion."""
from pathlib import Path
import sys
import time
import threading
import math
import numpy as np
from .contract import vector


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
    def __init__(self, adapter_root='/home/flyfuture/g2_hinge_assembly', timeout_s=2.):
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
        if gdk.gdk_init() != gdk.GDKRes.kSuccess:
            raise RuntimeError('GDK initialization failed')
        self.closed = False
        try:
            self.robot = gdk.Robot()
            self.controller = G2Controller(gdk, self.robot, allow_motion=False)
            self.streams = {'left_wrist': gdk.CameraType.kHandLeftColor,
                            'right_aux': gdk.CameraType.kHandRightColor}
            self.camera = gdk.Camera(list(self.streams.values()))
            time.sleep(1)
        except Exception:
            self.close()
            raise

    def observe(self):
        if self.closed:
            raise RuntimeError('Reader is closed')
        start = time.monotonic()
        self.controller.checked_arm_state()
        pose = self.controller.read_end_effector_pose('arm_l_end_link')
        state_received = time.monotonic_ns()
        state = np.asarray((*pose.position_m, *pose.orientation_xyzw), dtype=np.float32)
        if not np.isfinite(state).all() or abs(np.linalg.norm(state[3:]) - 1) > .01:
            raise ValueError('Invalid measured end-link pose')
        obs = {'state': state}
        stamps = {}
        for key, stream in self.streams.items():
            deadline = time.monotonic() + self.timeout_s
            while True:
                frame = self.camera.get_latest_image(stream, 100.)
                stamp = int(frame.timestamp_ns)
                if stamp > self.last_stamps.get(key, 0):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'No advancing camera timestamp: {key}')
                time.sleep(.005)
            obs[key] = self.decode(frame, self.gdk)
            stamps[key] = stamp
        self.last_stamps = stamps
        self.last_info = {'camera_timestamp_ns': stamps,
                          'state_received_monotonic_ns': state_received,
                          'received_monotonic_ns': time.monotonic_ns(),
                          'read_duration_s': time.monotonic() - start,
                          'motion_status': self.controller.motion_status_summary(),
                          'backend': 'gdk_read_only'}
        return obs

    def close(self):
        if not self.closed:
            self.closed = True
            self.camera = None
            self.controller = None
            self.robot = None
            self.gdk.gdk_release()
