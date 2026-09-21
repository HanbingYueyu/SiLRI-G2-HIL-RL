import time
import threading
import numpy as np
import pytest
from g2_local.config import LocalTaskConfig
from g2_local.contract import EpisodeContext
from g2_local.env import G2LocalEnv


class Reader:
    def __init__(self):
        self.state = np.array([.995,0,0,0,0,0,1], dtype=np.float64)
        self.closed = False
        self.fail = False
    def observe(self):
        if self.fail:
            raise OSError('camera disconnected')
        self.last_info = {'captured': time.monotonic()}
        return dict(state=self.state.copy(), left_wrist=np.zeros((16,16,3),dtype=np.uint8),
                    right_aux=np.zeros((16,16,3),dtype=np.uint8))
    def close(self):
        self.closed = True


class Port:
    def __init__(self, reader):
        self.reader = reader
        self.sent = []
        self.stopped = False
    def send(self, target):
        self.sent.append(target)
        self.reader.state = np.array((*target.position_m,*target.orientation_xyzw))
    def stop(self):
        self.stopped = True


def backend(reader, port, **kwargs):
    from g2_local.motion_backend import MotionBackend
    config = LocalTaskConfig(action_scale=(.01,)*6, workspace_low=(-1,)*3,workspace_high=(1,)*3)
    return MotionBackend(reader, port, config=config,
                         observation_guard=lambda obs, info, after: info['captured'] >= (after or 0),
                         outcome=lambda obs: (0., False), command_timeout=.3, send_timeout=.1,
                         step_period=.02, **kwargs)


def test_default_denies_motion():
    reader = Reader()
    port = Port(reader)
    driver = backend(reader, port)
    try:
        with pytest.raises(PermissionError):
            driver.execute((1,)*6)
        assert port.sent == []
    finally:
        driver.close()


def test_gym_acknowledged_effective_action_and_terminal_stop():
    reader = Reader()
    port = Port(reader)
    driver = backend(reader, port, allow_motion=True)
    env = G2LocalEnv(driver, max_steps=1)
    try:
        env.reset(options={'context': EpisodeContext('test',(0,0,0),'fixture','fixture')})
        obs, _, done, truncated, info = env.step(np.array([1,0,0,0,0,0]))
        assert not done and truncated
        assert info['executed_action'] == pytest.approx([.5,0,0,0,0,0])
        assert obs['state'][0] == pytest.approx(1.)
        assert port.stopped
        with pytest.raises(RuntimeError):
            driver.execute((0,)*6)
    finally:
        env.close()
    assert reader.closed


def test_successor_observation_failure_discards_step_and_stops():
    reader = Reader()
    class FailingPort(Port):
        def send(self, target):
            super().send(target)
            reader.fail = True
    port = FailingPort(reader)
    driver = backend(reader, port, allow_motion=True)
    try:
        with pytest.raises(OSError, match='camera'):
            driver.execute((0,)*6)
        assert port.stopped
    finally:
        driver.close()


def test_stale_observation_guard_blocks_before_send():
    reader = Reader()
    port = Port(reader)
    driver = backend(reader, port, allow_motion=True)
    driver.observation_guard = lambda *args: False
    try:
        with pytest.raises(RuntimeError, match='observation'):
            driver.execute((0,)*6)
        assert port.sent == []
    finally:
        driver.close()


def test_slow_camera_does_not_block_writer_but_lease_ends_step():
    reader = Reader()
    port = Port(reader)
    original = reader.observe
    reads = [0]
    def slow_observe():
        reads[0] += 1
        if reads[0] == 2:
            time.sleep(.4)
        return original()
    reader.observe = slow_observe
    driver = backend(reader, port, allow_motion=True)
    try:
        with pytest.raises(RuntimeError, match='lease'):
            driver.execute((0,)*6)
        assert len(port.sent) > 2
        assert port.stopped
    finally:
        driver.close()


def test_close_does_not_release_reader_under_blocked_sdk():
    reader = Reader()
    entered, unblock, stopped = threading.Event(), threading.Event(), threading.Event()
    class BlockingPort(Port):
        def send(self, target):
            entered.set()
            unblock.wait(2.)
        def stop(self):
            super().stop()
            stopped.set()
    port = BlockingPort(reader)
    driver = backend(reader, port, allow_motion=True, stop_timeout=.03)
    errors = []
    def execute():
        try:
            driver.execute((0,)*6)
        except Exception as exc:
            errors.append(exc)
    worker = threading.Thread(target=execute)
    worker.start()
    try:
        assert entered.wait(.5)
        with pytest.raises(TimeoutError, match='in flight'):
            driver.close()
        assert not reader.closed
    finally:
        unblock.set()
        worker.join(1.)
        assert stopped.wait(.5)
        driver.close()
    assert errors
    assert reader.closed


def test_close_does_not_release_reader_during_observation():
    reader = Reader()
    entered, unblock = threading.Event(), threading.Event()
    original = reader.observe
    def blocked_observe():
        entered.set()
        unblock.wait(2.)
        return original()
    reader.observe = blocked_observe
    driver = backend(reader, Port(reader), allow_motion=True, stop_timeout=.03)
    def observe():
        try:
            driver.observe()
        except RuntimeError:
            pass
    worker = threading.Thread(target=observe)
    worker.start()
    try:
        assert entered.wait(.5)
        with pytest.raises(TimeoutError, match='observation'):
            driver.close()
        assert not reader.closed
    finally:
        unblock.set()
        worker.join(1.)
        driver.close()
    assert reader.closed
