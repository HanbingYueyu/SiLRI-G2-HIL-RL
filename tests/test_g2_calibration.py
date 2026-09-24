from types import SimpleNamespace
import pytest
from g2_local import spacemouse


def make_frame(now, z=0., left=True, x=0.):
    return SimpleNamespace(axes=(0., 0., 0., x, 0., z), buttons=(left, False),
                           axis_times=(now, now), ready=True)


def test_yaw_requires_both_directions_neutral_and_release():
    session = spacemouse.RotationCheck(axis='yaw', started_at=0.)
    assert session.update(make_frame(1., z=-.8), now=1.)['stage'] == 'neutral'
    assert session.update(make_frame(2.), now=2.)['stage'] == 'positive'
    assert session.update(make_frame(2.1, z=-.8, x=.8), now=2.1)['stage'] == 'positive'
    assert session.update(make_frame(2.2, z=-.8), now=2.2)['stage'] == 'return_positive'
    assert session.update(make_frame(2.3, z=.8), now=2.3)['stage'] == 'return_positive'
    assert session.update(make_frame(2.4), now=2.4)['stage'] == 'negative'
    assert session.update(make_frame(2.5, z=.8), now=2.5)['stage'] == 'return_negative'
    assert session.update(make_frame(2.6), now=2.6)['stage'] == 'release'
    assert session.update(make_frame(2.7, left=False), now=2.7)['stage'] == 'passed'


def test_no_input_times_out_instead_of_passing():
    session = spacemouse.RotationCheck(axis='roll', started_at=0., timeout=2.)
    frame = make_frame(0., left=False)
    frame.ready = False
    assert session.update(frame, now=1.)['stage'] == 'press'
    assert session.update(frame, now=3.)['stage'] == 'failed'
    assert session.update(make_frame(3.1), now=3.1)['stage'] == 'failed'


def test_stale_neutral_cannot_advance_and_early_release_fails():
    session = spacemouse.RotationCheck(axis='yaw', started_at=0.)
    session.update(make_frame(1.), now=1.)
    assert session.update(make_frame(1.), now=2.)['stage'] == 'neutral'
    assert session.update(make_frame(2.1, left=False), now=2.1)['stage'] == 'failed'


@pytest.mark.parametrize('axis', ['yaw', 'pitch', 'roll'])
def test_fault_is_terminal(axis):
    session = spacemouse.RotationCheck(axis=axis, started_at=0.)
    assert session.fail('device disconnected')['stage'] == 'failed'
    assert session.update(make_frame(1.), now=1.)['stage'] == 'failed'


def test_runner_records_disconnect_and_stops():
    from g2_local import calibration
    class Reader:
        def poll(self):
            raise OSError('unplugged')
    output = []
    session = spacemouse.RotationCheck(axis='yaw', started_at=0.)
    with pytest.raises(OSError):
        calibration.run_check(Reader(), session, output.append)
    assert output[-1]['stage'] == 'failed'
    assert output[-1]['reason'] == 'reader_or_recording_error'


def test_runner_records_raw_evidence_for_success():
    from g2_local import calibration
    frames = iter([make_frame(1.), make_frame(1.01), make_frame(1.02, z=-.8),
                   make_frame(1.03), make_frame(1.04, z=.8),
                   make_frame(1.05), make_frame(1.06, left=False)])
    class Reader:
        def poll(self):
            self.current = next(frames)
            return self.current
    reader = Reader()
    output = []
    session = spacemouse.RotationCheck(axis='yaw', started_at=0.)
    calibration.run_check(reader, session, output.append,
                          clock=lambda: reader.current.axis_times[0], sleep=lambda _: None)
    assert output[-1]['stage'] == 'passed'
    assert output[2]['raw_axes'][5] == -.8
    assert output[4]['raw_axes'][5] == .8
