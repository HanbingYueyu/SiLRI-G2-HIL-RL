import time
import threading
import numpy as np
import pytest
from g2_local.config import LocalTaskConfig
from g2_local.contract import EpisodeContext
from g2_local.env import G2LocalEnv
from g2_local.freshness import FreshnessDecision


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
        self.stop_calls = 0
    def send(self, target):
        self.sent.append(target)
        self.reader.state = np.array((*target.position_m,*target.orientation_xyzw))
    def stop(self):
        self.stop_calls += 1
        self.stopped = True


@pytest.mark.parametrize('code,retry', [('mapping_warming_up',True),
                                      ('mapping_expired',True),
                                      ('mapping_unavailable',False)])
def test_mapping_startup_wait_is_transient_only(code, retry):
    reader = Reader()
    port = Port(reader)
    class Guard:
        calls = 0
        last_decision = FreshnessDecision(code, 'fixture')
        def __call__(self, *args, **kwargs):
            self.calls += 1
            return self.calls == 2
    guard = Guard()
    driver = backend(reader,port,observation_guard=guard,startup_mapping_wait_s=1.)
    try:
        if retry:
            driver.observe()
            with pytest.raises(RuntimeError, match=code):
                driver.observe()
            assert guard.calls == 3
        else:
            with pytest.raises(RuntimeError, match=code):
                driver.observe()
            assert guard.calls == 1
        assert not port.sent
    finally:
        driver.close()


def test_startup_discards_old_camera_without_sending_then_never_retries_live():
    reader = Reader()
    port = Port(reader)
    class Guard:
        calls = 0
        last_decision = FreshnessDecision('camera_stale:left_wrist', 'old startup frame')
        def __call__(self, *args, **kwargs):
            self.calls += 1
            return self.calls == 2
    guard = Guard()
    driver = backend(reader, port, observation_guard=guard, allow_motion=True,
                     startup_camera_wait_s=2.)
    try:
        driver.observe()
        assert guard.calls == 2 and not port.sent
        with pytest.raises(RuntimeError, match='camera_stale'):
            driver.observe()
        assert guard.calls == 3
    finally:
        driver.close()


def test_bound_method_preserves_freshness_rejection_code():
    reader = Reader()
    port = Port(reader)
    class Lease:
        last_decision = FreshnessDecision('camera_stale:left_wrist', 'old camera')
        def accept(self, *args, **kwargs):
            return False
    driver = backend(reader, port, observation_guard=Lease().accept, allow_motion=True)
    try:
        with pytest.raises(RuntimeError, match='camera_stale:left_wrist'):
            driver.observe()
    finally:
        driver.close()


def test_precommand_pose_read_does_not_wait_for_another_camera_frame():
    reader = Reader()
    reader.state[0] = .5
    port = Port(reader)
    calls = []
    original = reader.observe
    def observe():
        calls.append('camera')
        return original()
    def control_pose():
        calls.append('pose')
        return reader.state.copy()
    reader.observe = observe
    reader.read_control_pose = control_pose
    def revalidate(obs, info):
        calls.append('revalidate')
        return True
    driver = backend(reader, port, allow_motion=True, reference_guard=revalidate)
    try:
        predecessor = driver.observe()
        reader.state[0] = .502
        driver.execute_from((1, 0, 0, 0, 0, 0), predecessor)
        assert calls == ['camera', 'pose', 'revalidate', 'camera']
        assert port.sent[0].position_m[0] == pytest.approx(.51)
    finally:
        driver.close()


def test_fast_pose_path_still_rejects_expired_predecessor():
    reader = Reader()
    reader.read_control_pose = lambda: reader.state.copy()
    port = Port(reader)
    driver = backend(reader, port, allow_motion=True, reference_guard=lambda *args: False)
    try:
        predecessor = driver.observe()
        with pytest.raises(RuntimeError, match='Policy input expired'):
            driver.execute_from((0.,)*6, predecessor)
        assert not port.sent
    finally:
        driver.close()


def backend(reader, port, *, observation_guard=None, outcome=None, **kwargs):
    from g2_local.motion_backend import MotionBackend
    config = LocalTaskConfig(action_scale=(.01,)*6, workspace_low=(-1,)*3,workspace_high=(1,)*3)
    if observation_guard is None:
        observation_guard = lambda obs, info, after, after_sdk_ns=None: info['captured'] >= (after or 0)
    if outcome is None:
        outcome = lambda obs: (0., False)
    return MotionBackend(reader, port, config=config,
                         observation_guard=observation_guard, outcome=outcome,
                         command_timeout=.3, send_timeout=.1,
                         step_period=.02, **kwargs)


class RejectingGuard:
    def __init__(self, code, *, after_send=False):
        self.code = code
        self.after_send = after_send
        self.healthy = False
        self.last_decision = FreshnessDecision('not_checked', 'fixture')

    def __call__(self, obs, info, after, after_sdk_ns=None):
        reject = not self.healthy and (after is not None if self.after_send else True)
        self.last_decision = FreshnessDecision(
            self.code if reject else 'ok', 'fixture decision')
        return not reject


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


def test_action_origin_is_policy_predecessor_not_new_feedback():
    reader = Reader()
    reader.state[0] = .5
    port = Port(reader)
    driver = backend(reader, port, allow_motion=True, reference_guard=lambda *args: True)
    try:
        predecessor = driver.observe()
        reader.state[0] = .502
        driver.execute_from((1,0,0,0,0,0), predecessor)
        assert port.sent[0].position_m[0] == pytest.approx(.51)
    finally:
        driver.close()


def test_active_command_uses_camera_budget_within_lease_while_startup_uses_reader_default():
    reader = Reader()
    timeouts = []
    reader.observe_with_timeout = lambda timeout_s: timeouts.append(timeout_s) or reader.observe()
    port = Port(reader)
    driver = backend(reader, port, allow_motion=True)
    try:
        driver.observe()
        assert timeouts == []
        driver.execute((0,)*6)
        assert timeouts == [.2]
    finally:
        driver.close()


def test_successor_waits_briefly_for_camera_frames_after_command_send():
    reader = Reader()
    port = Port(reader)

    class Guard:
        def __init__(self):
            self.after_calls = 0
            self.last_decision = FreshnessDecision('ok', 'fixture')

        def __call__(self, _obs, _info, after, after_sdk_ns=None):
            if after is None:
                return True
            self.after_calls += 1
            if self.after_calls == 1:
                self.last_decision = FreshnessDecision(
                    'not_after_command:left_wrist', 'frame predates command')
                return False
            self.last_decision = FreshnessDecision('ok', 'new frame')
            return True

    guard = Guard()
    driver = backend(reader, port, observation_guard=guard, allow_motion=True)
    try:
        result = driver.execute((0,)*6)
        assert result.observation['state'].shape == (7,)
        assert guard.after_calls == 2
        assert len(port.sent) > 0
    finally:
        driver.close()


def test_missing_post_command_camera_frame_uses_recoverable_camera_fault():
    from g2_local.gdk_backend import CameraUnavailable

    reader = Reader()
    port = Port(reader)

    class Guard:
        last_decision = FreshnessDecision('not_after_command:left_wrist', 'frame predates command')

        def __call__(self, _obs, _info, after, after_sdk_ns=None):
            return after is None

    driver = backend(reader, port, observation_guard=Guard(), allow_motion=True)
    try:
        with pytest.raises(CameraUnavailable, match='post-command camera frame'):
            driver.execute((0,)*6)
        assert driver.stop_confirmed is True
    finally:
        driver.close()


def test_expired_policy_predecessor_cannot_send():
    reader = Reader()
    port = Port(reader)
    driver = backend(reader, port, allow_motion=True, reference_guard=lambda *args: False)
    try:
        predecessor = driver.observe()
        with pytest.raises(RuntimeError, match='Policy input expired'):
            driver.execute_from((0,)*6, predecessor)
        assert port.sent == []
    finally:
        driver.close()


def test_final_terminal_poll_cancels_before_command_submission():
    from g2_local.real_episode import TerminalBeforeCommand
    reader = Reader()
    port = Port(reader)
    def terminal():
        raise TerminalBeforeCommand('success', time.monotonic_ns())
    driver = backend(reader, port, allow_motion=True, before_command=terminal)
    try:
        with pytest.raises(TerminalBeforeCommand):
            driver.execute((1,0,0,0,0,0))
        assert not port.sent and driver.stopped
    finally:
        driver.close()


def test_episode_local_envelope_rejects_target_before_send():
    from g2_local.local_envelope import LocalEnvelope
    reader = Reader()
    reader.state[0] = .5
    port = Port(reader)
    driver = backend(reader, port, allow_motion=True,
                     local_envelope=LocalEnvelope((-.005,)*3, (.005,)*3, .1))
    env = G2LocalEnv(driver)
    try:
        env.reset(options={'context': EpisodeContext('local',(0,0,0),'fixture','fixture')})
        assert driver.episode_reference[0] == .5
        with pytest.raises(ValueError, match='translation envelope'):
            env.step(np.array([1,0,0,0,0,0]))
        assert port.sent == []
        assert driver.stopped
    finally:
        env.close()


def test_episode_local_envelope_rejects_missing_reference_and_measured_escape():
    from g2_local.local_envelope import LocalEnvelope
    for missing in (True, False):
        reader = Reader()
        reader.state[0] = .5
        port = Port(reader)
        driver = backend(reader, port, allow_motion=True,
                         local_envelope=LocalEnvelope((-.005,)*3, (.005,)*3, .1))
        try:
            if not missing:
                driver.begin_episode(driver.observe())
                reader.state[0] += .02
            with pytest.raises((RuntimeError, ValueError)):
                driver.execute((0,)*6)
            assert port.sent == []
            assert driver.stopped
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


def test_mapping_failure_before_plan_sends_nothing_and_stops_backend():
    reader = Reader()
    port = Port(reader)
    guard = RejectingGuard('mapping_expired')
    driver = backend(reader, port, observation_guard=guard, allow_motion=True)
    try:
        with pytest.raises(RuntimeError, match='mapping_expired'):
            driver.execute((0,)*6)
        assert port.sent == []
        assert driver.stopped is True
        guard.healthy = True
        with pytest.raises(RuntimeError, match='reconstruct'):
            driver.observe()
    finally:
        driver.close()


def test_ambiguous_successor_stops_once_discards_step_and_never_auto_recovers():
    reader = Reader()
    port = Port(reader)
    guard = RejectingGuard('not_after_command:tf', after_send=True)
    outcomes = []
    driver = backend(reader, port, observation_guard=guard,
                     outcome=lambda obs: outcomes.append(obs) or (0., False),
                     allow_motion=True)
    try:
        with pytest.raises(RuntimeError, match='not_after_command:tf'):
            driver.execute((0,)*6)
        assert port.stop_calls == 1
        assert outcomes == []
        assert driver.stopped is True
        guard.healthy = True
        with pytest.raises(RuntimeError, match='reconstruct'):
            driver.observe()
        assert port.stop_calls == 1
    finally:
        driver.close()


@pytest.mark.parametrize('result', [False, None, 1, np.bool_(True)])
def test_guard_accepts_only_the_exact_true_singleton(result):
    reader = Reader()
    port = Port(reader)

    class Guard:
        last_decision = FreshnessDecision('mapping_invalid', 'fixture')

        def __call__(self, obs, info, after, after_sdk_ns=None):
            return result

    driver = backend(reader, port, observation_guard=Guard(), allow_motion=True)
    try:
        with pytest.raises(RuntimeError, match='mapping_invalid'):
            driver.execute((0,)*6)
        assert port.sent == []
        assert driver.stopped is True
    finally:
        driver.close()


def test_diagnostic_lookup_failure_still_fails_closed_without_masking_rejection():
    reader = Reader()
    port = Port(reader)

    class Guard:
        diagnostic_reads = 0

        def __call__(self, obs, info, after, after_sdk_ns=None):
            return False

        @property
        def last_decision(self):
            self.diagnostic_reads += 1
            raise LookupError('diagnostic unavailable')

    guard = Guard()
    driver = backend(reader, port, observation_guard=guard, allow_motion=True)
    try:
        with pytest.raises(RuntimeError, match='freshness not confirmed'):
            driver.execute((0,)*6)
        assert guard.diagnostic_reads == 1
        assert port.sent == []
        assert driver.stopped is True
    finally:
        driver.close()


def test_unsafe_diagnostic_code_is_not_interpolated_into_the_error():
    reader = Reader()
    port = Port(reader)

    class Guard:
        last_decision = FreshnessDecision('unsafe\noperator message', 'fixture')

        def __call__(self, obs, info, after, after_sdk_ns=None):
            return False

    driver = backend(reader, port, observation_guard=Guard(), allow_motion=True)
    try:
        with pytest.raises(RuntimeError) as caught:
            driver.execute((0,)*6)
        assert str(caught.value) == 'Source observation freshness not confirmed'
        assert port.sent == []
        assert driver.stopped is True
    finally:
        driver.close()


def test_guard_exception_fails_closed_before_send():
    reader = Reader()
    port = Port(reader)

    def broken_guard(obs, info, after, after_sdk_ns=None):
        raise TimeoutError('snapshot timed out')

    driver = backend(reader, port, observation_guard=broken_guard, allow_motion=True)
    try:
        with pytest.raises(TimeoutError, match='snapshot timed out'):
            driver.execute((0,)*6)
        assert port.sent == []
        assert driver.stopped is True
    finally:
        driver.close()


def test_slow_camera_does_not_block_writer_but_lease_ends_step():
    from g2_local.gdk_backend import CameraUnavailable

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
        with pytest.raises(CameraUnavailable, match='command lease'):
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
