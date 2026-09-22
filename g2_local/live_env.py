"""Safe live G2 Gym construction with an explicit read-only default.

The factory in this module is intentionally narrower than the motion path: it
creates the clock client, freshness guard, and GDK reader, but it does not
construct a command port.  A caller must explicitly choose a different,
commissioned motion integration before any command boundary exists.
"""
from copy import deepcopy
from pathlib import Path
import logging

from .clock_ipc import SnapshotClient
from .config import HingeInsertTaskConfig
from .env import G2LocalEnv
from .freshness import FreshnessLimits, ObservationFreshnessGuard
from .gdk_backend import GdkReader


def _decision_code(guard):
    try:
        code = guard.last_decision.code
    except Exception:
        return None
    if (type(code) is str and 0 < len(code) <= 128 and code.isascii() and
            all(character.isalnum() or character in '_:-.' for character in code)):
        return code
    return None


class ReadOnlyG2Backend:
    """Gym backend that validates live observations but can never command motion."""

    name = 'gdk_read_only'

    def __init__(self, reader, client, observation_guard):
        if not callable(getattr(reader, 'observe', None)):
            raise ValueError('A GDK reader is required')
        if not callable(getattr(reader, 'close', None)):
            raise ValueError('A closable GDK reader is required')
        if not callable(getattr(client, 'close', None)):
            raise ValueError('A closable clock client is required')
        if not callable(observation_guard):
            raise ValueError('An observation freshness guard is required')
        self.reader = reader
        self.client = client
        self.observation_guard = observation_guard
        self.closed = False

    def observe(self):
        if self.closed:
            raise RuntimeError('Read-only G2 backend is closed')
        observation = self.reader.observe()
        info = deepcopy(getattr(self.reader, 'last_info', {}))
        if self.observation_guard(observation, info, None) is not True:
            code = _decision_code(self.observation_guard)
            detail = 'Source observation freshness not confirmed'
            if code is not None:
                detail += f': {code}'
            raise RuntimeError(detail)
        return observation

    def execute(self, action):
        del action
        raise PermissionError(
            'G2 environment is read-only; no command port exists and no motion was sent')

    def stop(self):
        # There is no command stream and therefore no hold/stop write to issue.
        return None

    def close(self):
        if self.closed:
            return
        self.closed = True
        reader_error = client_error = None
        try:
            self.reader.close()
        except Exception as error:
            reader_error = error
            logging.exception('GDK reader cleanup failed')
        try:
            self.client.close()
        except Exception as error:
            client_error = error
            logging.exception('Clock client cleanup failed')
        if reader_error is not None:
            raise reader_error
        if client_error is not None:
            raise client_error


def create_g2_env(*, clock_socket, expected_master, limits: FreshnessLimits,
                  task: HingeInsertTaskConfig | None = None,
                  adapter_root='/home/flyfuture/g2_hinge_assembly',
                  reader_timeout_s=2., clock_timeout_s=.25,
                  image_size=128, max_steps=None,
                  reader_factory=None, client_factory=None, guard_factory=None):
    """Create a live dual-camera Gym environment in read-only mode.

    ``limits`` and ``expected_master`` are mandatory by design.  The default
    path creates no ``GdkCommandPort`` and therefore cannot send a robot
    command.  Factories are test seams and must return the same read-only
    interfaces as the production classes.
    """
    if type(limits) is not FreshnessLimits:
        raise ValueError('Explicit FreshnessLimits are required')
    if type(expected_master) is not str or not expected_master.strip():
        raise ValueError('Expected PTP master identity is required')
    if task is None:
        task = HingeInsertTaskConfig()
    if not isinstance(task, HingeInsertTaskConfig):
        raise ValueError('task must be HingeInsertTaskConfig')
    if max_steps is None:
        max_steps = task.max_episode_steps
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError('max_steps must be a positive integer')
    if type(image_size) is not int or image_size <= 0:
        raise ValueError('image_size must be a positive integer')

    make_client = client_factory or (
        lambda **kwargs: SnapshotClient(
            Path(kwargs['clock_socket']), timeout_s=kwargs['timeout_s'],
            expected_master=kwargs['expected_master']))
    make_reader = reader_factory or (
        lambda **kwargs: GdkReader(adapter_root=kwargs['adapter_root'],
                                   timeout_s=kwargs['timeout_s']))
    make_guard = guard_factory or (lambda client, limits:
                                   ObservationFreshnessGuard(client, limits))

    client = reader = backend = None
    try:
        client = make_client(clock_socket=clock_socket, expected_master=expected_master,
                             timeout_s=clock_timeout_s)
        reader = make_reader(adapter_root=adapter_root, timeout_s=reader_timeout_s)
        guard = make_guard(client, limits)
        backend = ReadOnlyG2Backend(reader, client, guard)
        return G2LocalEnv(backend, max_steps=max_steps, image_size=image_size)
    except BaseException:
        if backend is not None:
            backend.close()
        else:
            for resource in (reader, client):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception:
                        logging.exception('Live G2 setup cleanup failed')
        raise
