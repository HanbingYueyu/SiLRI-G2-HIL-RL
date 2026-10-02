"""The sole assembly path for commissioned live GDK motion."""

from dataclasses import dataclass
from typing import Callable

from .env import G2LocalEnv
from .freshness import FreshnessLeaseGuard, ObservationFreshnessGuard
from .motion_backend import MotionBackend


@dataclass(frozen=True)
class MotionFactories:
    clock_client: Callable
    reader: Callable
    command_port: Callable

    @staticmethod
    def production():
        # The SDK import stays behind the complete permission and evidence gates.
        from .gdk_backend import GdkCommandPort, GdkReader

        def command_port(controller, **kwargs):
            from .safe_stop import prepare_safe_stop
            request = prepare_safe_stop(controller.robot, kwargs['expected_mode'])
            return GdkCommandPort(controller, safe_stop_request=request, **kwargs)

        return MotionFactories(_LocalClock, GdkReader, command_port)


class OwnedObservationSource:
    """One lifetime for a GDK reader and its previously constructed clock client."""

    def __init__(self, reader, clock_client):
        self.reader = reader
        self.clock_client = clock_client
        self._closed = False

    @property
    def controller(self):
        return self.reader.controller

    @property
    def last_info(self):
        return self.reader.last_info

    def observe(self):
        return self.reader.observe()

    def observe_with_timeout(self, timeout_s):
        bounded_observe = getattr(self.reader, 'observe_with_timeout', None)
        if callable(bounded_observe):
            return bounded_observe(timeout_s)
        return self.reader.observe()

    @property
    def read_control_pose(self):
        return getattr(self.reader, 'read_control_pose', None)

    def read_sdk_clock_ns(self):
        """Delegate the command-send clock anchor; absent readers raise."""
        source = getattr(self.reader, 'read_sdk_clock_ns', None)
        if not callable(source):
            raise AttributeError('Reader exposes no SDK clock')
        return source()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.reader.close()
        finally:
            self.clock_client.close()


class _LocalClock:
    """Closable placeholder: freshness comes from local receipt times.

    Upstream G2 flow never converts robot timestamps into local time; it asks
    how long this process has gone without a new frame.  Real time
    synchronisation stays the operator's job (`ptp_hard.sh`), used for logging
    and the GDK latency APIs, and is not needed for this check.
    """

    def __init__(self, *_args, **_kwargs):
        # Used through the MotionFactories clock_client seam; it accepts and
        # ignores the legacy (socket, master) arguments.
        return None

    def close(self):
        return None


def _preflight_mode(controller, expected_mode):
    if type(expected_mode) is not int or expected_mode not in (1, 3):
        raise ValueError('Explicit control mode 1 or 3 required')
    controller.checked_arm_state()
    status = controller.motion_status_summary()
    if (type(status) is not dict or type(status.get('control_mode')) is not int or
            type(status.get('error_code')) is not int or status['error_code'] != 0):
        raise RuntimeError(f'Unhealthy or changed control mode: {status}')
    if status['control_mode'] == expected_mode:
        return
    # The upstream visual flow may leave GDK in joint impedance mode 3.
    # For a commissioned position-mode session, use its existing verified
    # takeover API before constructing any command port or publishing targets.
    if status['control_mode'] == 3 and expected_mode == 1:
        switch = getattr(controller, 'enter_position_control', None)
        if callable(switch):
            print('检测到 control_mode=3；切换到 SiLRI 配置要求的 control_mode=1。', flush=True)
            switch('SiLRI supervised motion startup')
            controller.checked_arm_state()
            status = controller.motion_status_summary()
            if (type(status) is dict and type(status.get('control_mode')) is int and
                    status['control_mode'] == expected_mode and
                    type(status.get('error_code')) is int and status['error_code'] == 0):
                return
    raise RuntimeError(f'Unhealthy or changed control mode: {status}')


def create_motion_env(config, coordinator, *, cli_allow_motion, factories=None,
                      skip_tf_progress=False, skip_state_progress=False) -> G2LocalEnv:
    if (type(cli_allow_motion) is not bool or cli_allow_motion is not True or
            config.requested_motion is not True or config.motion_permitted is not True):
        raise PermissionError('commissioned motion permission is required')
    if type(skip_tf_progress) is not bool:
        raise ValueError('skip_tf_progress must be boolean')
    if config.commissioning.verify_files_and_hashes(config.freshness) is not True:
        raise PermissionError('commissioning evidence is not current')
    config.motion.limits.validate_motion()
    if type(config.motion.control_mode) is not int or config.motion.control_mode not in (1, 3):
        raise ValueError('Explicit control mode 1 or 3 required')
    factories = MotionFactories.production() if factories is None else factories

    # Freshness is local (see _LocalClock): nothing reads a clock in this
    # process.  The legacy client factory stays only as the lifetime seam that
    # owns reader/client teardown together.
    client = factories.clock_client(None, None)
    source = None
    port = None
    backend = None
    try:
        reader = factories.reader(adapter_root=config.motion.adapter_root,
                                  timeout_s=config.motion.reader_timeout_s,
                                  allow_motion=True)
        source = OwnedObservationSource(reader, client)
        _preflight_mode(source.controller, config.motion.control_mode)
        observation_guard = ObservationFreshnessGuard(
            config.freshness, skip_tf_progress=skip_tf_progress,
            skip_state_progress=skip_state_progress)
        lease = FreshnessLeaseGuard(observation_guard,
                                    feedback_lease_s=(config.motion.command_timeout_s +
                                                      config.motion.send_timeout_s))
        stop_options = {}
        if callable(getattr(reader, 'read_stop_feedback', None)):
            stop_guard = ObservationFreshnessGuard(
                config.freshness, skip_state_progress=skip_state_progress)
            def stop_pose_provider():
                # Do not race an SDK read or release its resources under a hold.
                if backend is None or not backend.reader_lock.acquire(
                        timeout=config.motion.stop_timeout_s):
                    raise TimeoutError('Independent stop feedback reader is busy')
                try:
                    info = reader.read_stop_feedback()
                    stop_guard.validate_stop_feedback(info)
                    from .motion_backend import PoseTarget
                    pose = info['motion_pose']
                    return PoseTarget(tuple(pose[:3]), tuple(pose[3:]))
                finally:
                    backend.reader_lock.release()
            stop_options['stop_pose_provider'] = stop_pose_provider
        port = factories.command_port(source.controller,
                                      expected_mode=config.motion.control_mode,
                                      allow_motion=True, freshness_guard=lease,
                                      life_time_s=config.motion.command_lifetime_s,
                                      **stop_options)
        backend = MotionBackend(source, port, config=config.motion.limits,
                                observation_guard=lease.accept,
                                outcome=coordinator.outcome,
                                command_timeout=config.motion.command_timeout_s,
                                send_timeout=config.motion.send_timeout_s,
                                stop_timeout=config.motion.stop_timeout_s,
                                send_rate_hz=config.motion.send_rate_hz,
                                local_envelope=config.motion.local_envelope,
                                reference_guard=lease.revalidate,
                                startup_camera_wait_s=2.,
                                startup_mapping_wait_s=60.,
                                policy_position_drift_m=config.motion.policy_position_drift_m,
                                policy_rotation_drift_rad=config.motion.policy_rotation_drift_rad,
                                before_command=getattr(coordinator, 'before_command', None),
                                step_period=1. / config.task.control_hz,
                                allow_motion=True)
        return G2LocalEnv(backend, max_steps=config.task.max_episode_steps,
                          image_size=config.observation.image_size,
                          camera_rois=config.observation.camera_rois,
                          intervention=coordinator.intervention)
    except BaseException:
        if backend is not None:
            backend.close()
        elif source is not None:
            try:
                if port is not None:
                    port.stop()
            finally:
                source.close()
        else:
            client.close()
        raise
