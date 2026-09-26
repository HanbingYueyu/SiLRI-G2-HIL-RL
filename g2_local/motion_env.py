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
        from .clock_ipc import SnapshotClient
        from .gdk_backend import GdkCommandPort, GdkReader

        def clock_client(socket, master):
            return SnapshotClient(socket, timeout_s=2., expected_master=master)

        return MotionFactories(clock_client, GdkReader, GdkCommandPort)


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

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.reader.close()
        finally:
            self.clock_client.close()


def _preflight_mode(controller, expected_mode):
    if type(expected_mode) is not int or expected_mode not in (1, 3):
        raise ValueError('Explicit control mode 1 or 3 required')
    controller.checked_arm_state()
    status = controller.motion_status_summary()
    if (type(status) is not dict or type(status.get('control_mode')) is not int or
            status['control_mode'] != expected_mode or
            type(status.get('error_code')) is not int or status['error_code'] != 0):
        raise RuntimeError(f'Unhealthy or changed control mode: {status}')


def create_motion_env(config, coordinator, *, cli_allow_motion, factories=None) -> G2LocalEnv:
    if (type(cli_allow_motion) is not bool or cli_allow_motion is not True or
            config.requested_motion is not True or config.motion_permitted is not True):
        raise PermissionError('commissioned motion permission is required')
    if config.commissioning.verify_files_and_hashes(config.freshness) is not True:
        raise PermissionError('commissioning evidence is not current')
    config.motion.limits.validate_motion()
    if type(config.motion.control_mode) is not int or config.motion.control_mode not in (1, 3):
        raise ValueError('Explicit control mode 1 or 3 required')
    factories = MotionFactories.production() if factories is None else factories

    client = factories.clock_client(config.commissioning.clock_socket,
                                    config.commissioning.expected_master)
    source = None
    port = None
    backend = None
    try:
        reader = factories.reader(adapter_root=config.motion.adapter_root,
                                  timeout_s=config.motion.reader_timeout_s,
                                  allow_motion=True)
        source = OwnedObservationSource(reader, client)
        _preflight_mode(source.controller, config.motion.control_mode)
        observation_guard = ObservationFreshnessGuard(client, config.freshness)
        lease = FreshnessLeaseGuard(observation_guard,
                                    feedback_lease_s=config.motion.command_timeout_s)
        port = factories.command_port(source.controller,
                                      expected_mode=config.motion.control_mode,
                                      allow_motion=True, freshness_guard=lease,
                                      life_time_s=config.motion.command_lifetime_s)
        backend = MotionBackend(source, port, config=config.motion.limits,
                                observation_guard=lease.accept,
                                outcome=coordinator.outcome,
                                command_timeout=config.motion.command_timeout_s,
                                send_timeout=config.motion.send_timeout_s,
                                stop_timeout=config.motion.stop_timeout_s,
                                send_rate_hz=config.motion.send_rate_hz,
                                local_envelope=config.motion.local_envelope,
                                reference_guard=observation_guard.revalidate,
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
