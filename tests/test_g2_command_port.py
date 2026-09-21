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
