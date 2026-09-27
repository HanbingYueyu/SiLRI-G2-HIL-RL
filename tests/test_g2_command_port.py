from types import SimpleNamespace
import pytest


class Controller:
    def __init__(self):
        self.sent = []
        self.mode = 1
        self.pose = SimpleNamespace(position_m=(.1,.2,.3), orientation_xyzw=(0,0,0,1))
    def checked_arm_state(self):
        return (0.,)*14
    def motion_status_summary(self):
        return {'control_mode': self.mode, 'error_code': 0}
    def read_end_effector_pose(self, _):
        return self.pose
    def _send_left_cartesian_pose(self, target, lifetime):
        self.sent.append((target,lifetime))


def test_sender_waiting_for_robot_lock_rechecks_watchdog_before_sending():
    import threading
    from g2_local.gdk_backend import GdkCommandPort, _SerializedRobot

    controller = Controller()
    controller.robot = _SerializedRobot(SimpleNamespace())
    port = GdkCommandPort(controller, expected_mode=1, allow_motion=True,
                          freshness_guard=lambda: True)
    cancelled = threading.Event()
    started = threading.Event()
    errors = []
    port.bind_cancel_event(cancelled)

    def send():
        started.set()
        try:
            port.send(controller.pose)
        except Exception as error:
            errors.append(error)

    with controller.robot._sdk_call_lock:
        thread = threading.Thread(target=send)
        thread.start()
        assert started.wait(1)
        cancelled.set()
    thread.join(2)
    assert not thread.is_alive()
    assert len(errors) == 1 and 'halted' in str(errors[0])
    assert controller.sent == []


def test_camera_permission_loss_uses_independently_verified_stop_pose():
    from g2_local.gdk_backend import GdkCommandPort
    controller = Controller()
    fresh = [True]
    port = GdkCommandPort(controller, expected_mode=1, allow_motion=True,
                          freshness_guard=lambda: fresh[0],
                          stop_pose_provider=lambda: controller.pose)
    port.send(controller.pose)
    fresh[0] = False
    port.stop()
    port.stop()
    assert len(controller.sent) == 2


def test_independent_stop_feedback_failure_does_not_send_hold():
    from g2_local.gdk_backend import GdkCommandPort
    controller = Controller()
    def unavailable():
        raise RuntimeError('state_stale:tf')
    port = GdkCommandPort(controller, expected_mode=1, allow_motion=True,
                          freshness_guard=lambda: True, stop_pose_provider=unavailable)
    port.send(controller.pose)
    with pytest.raises(RuntimeError, match='physical stop unconfirmed'):
        port.stop()
    assert len(controller.sent) == 1


def test_expired_mapping_requests_independent_safe_stop_once_but_not_confirmation():
    from g2_local.gdk_backend import GdkCommandPort
    controller = Controller()
    calls = []
    def expired():
        raise RuntimeError('mapping_expired')
    port = GdkCommandPort(controller, expected_mode=1, allow_motion=True,
                          freshness_guard=lambda: True, stop_pose_provider=expired,
                          safe_stop_request=lambda: calls.append('left') or 0)
    port.send(controller.pose)
    for _ in range(2):
        with pytest.raises(RuntimeError, match='physical stop unconfirmed'):
            port.stop()
    assert calls == ['left']
    assert len(controller.sent) == 1
    assert port.safe_stop_acknowledged is True
    with pytest.raises(RuntimeError, match='stopped'):
        port.send(controller.pose)


@pytest.mark.parametrize('reply', [None, False, 1])
def test_safe_stop_invalid_reply_never_counts_as_acknowledgement(reply):
    from g2_local.gdk_backend import GdkCommandPort
    controller = Controller()
    def expired():
        raise RuntimeError('mapping_expired')
    port = GdkCommandPort(controller, expected_mode=1, allow_motion=True,
                          freshness_guard=lambda: True, stop_pose_provider=expired,
                          safe_stop_request=lambda: reply)
    port.send(controller.pose)
    with pytest.raises(RuntimeError, match='physical stop unconfirmed'):
        port.stop()
    assert port.safe_stop_acknowledged is False
    assert port.safe_stop_error is not None


def test_disabled_port_never_sends_even_stop():
    from g2_local.gdk_backend import GdkCommandPort
    controller = Controller()
    port = GdkCommandPort(controller, expected_mode=1)
    with pytest.raises(PermissionError):
        port.send(controller.pose)
    port.stop()
    assert controller.sent == []


def test_stop_holds_current_measured_pose_once_and_latches():
    from g2_local.gdk_backend import GdkCommandPort
    controller = Controller()
    port = GdkCommandPort(controller, expected_mode=1, allow_motion=True,
                          freshness_guard=lambda: True)
    initial = controller.pose
    port.send(initial)
    controller.pose = SimpleNamespace(position_m=(.2,.3,.4), orientation_xyzw=(0,0,0,1))
    port.stop()
    port.stop()
    assert len(controller.sent) == 2
    assert controller.sent[-1][0] is controller.pose
    with pytest.raises(RuntimeError):
        port.send(initial)


def test_mode_fault_no_hold_command_and_no_resume():
    from g2_local.gdk_backend import GdkCommandPort
    controller = Controller()
    port = GdkCommandPort(controller, expected_mode=1, allow_motion=True,
                          freshness_guard=lambda: True)
    port.send(controller.pose)
    controller.mode = 3
    with pytest.raises(RuntimeError):
        port.stop()
    assert len(controller.sent) == 1
    controller.mode = 1
    with pytest.raises(RuntimeError):
        port.send(controller.pose)


def test_enabling_requires_explicit_freshness_guard():
    from g2_local.gdk_backend import GdkCommandPort
    with pytest.raises(ValueError):
        GdkCommandPort(Controller(), expected_mode=1, allow_motion=True)


@pytest.mark.parametrize('result', [False, None])
def test_guard_must_explicitly_confirm_freshness(result):
    from g2_local.gdk_backend import GdkCommandPort
    controller = Controller()
    port = GdkCommandPort(controller, expected_mode=1, allow_motion=True,
                          freshness_guard=lambda: result)
    with pytest.raises(RuntimeError, match='freshness'):
        port.send(controller.pose)
    assert controller.sent == []
