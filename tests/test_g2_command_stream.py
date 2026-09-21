import threading
import time
import pytest


class Port:
    def __init__(self):
        self.sent = []
        self.stopped = threading.Event()
    def send(self, target):
        self.sent.append((time.monotonic(), target))
    def stop(self):
        self.stopped.set()


def test_repeats_target_acknowledges_and_lease_expires():
    from g2_local.command_stream import CommandStream
    port = Port()
    stream = CommandStream(port, command_timeout=.15, send_timeout=.1)
    seq = stream.submit((1,2,3))
    assert stream.wait_sent(seq, timeout=.3) > 0
    assert port.stopped.wait(1.)
    assert len(port.sent) >= 2
    with pytest.raises(RuntimeError, match='lease'):
        stream.check()
    with pytest.raises(RuntimeError):
        stream.submit((4,5,6))
    stream.stop()
    count = len(port.sent)
    time.sleep(.03)
    assert len(port.sent) == count


def test_blocking_send_watchdog_and_bounded_stop():
    from g2_local.command_stream import CommandStream
    unblock, entered = threading.Event(), threading.Event()
    class BlockingPort(Port):
        def send(self, target):
            entered.set()
            unblock.wait(2.)
            super().send(target)
    port = BlockingPort()
    stream = CommandStream(port, command_timeout=.5, send_timeout=.05, stop_timeout=.03)
    try:
        seq = stream.submit((1,))
        assert entered.wait(.5)
        with pytest.raises(RuntimeError, match='send'):
            stream.wait_sent(seq, timeout=.5)
        with pytest.raises(TimeoutError, match='in flight'):
            stream.stop()
        assert not port.stopped.is_set()
    finally:
        unblock.set()
        assert port.stopped.wait(1.)
        stream.stop()


def test_stop_before_first_command_never_sends():
    from g2_local.command_stream import CommandStream
    port = Port()
    stream = CommandStream(port, command_timeout=.2, send_timeout=.1)
    stream.stop()
    with pytest.raises(RuntimeError):
        stream.submit((1,))
    assert port.sent == []


def test_sender_failure_propagates_and_stops():
    from g2_local.command_stream import CommandStream
    class BrokenPort(Port):
        def send(self, target):
            raise OSError('SDK rejected')
    port = BrokenPort()
    stream = CommandStream(port, command_timeout=.2, send_timeout=.1)
    seq = stream.submit((1,))
    with pytest.raises(RuntimeError, match='SDK rejected'):
        stream.wait_sent(seq, timeout=.5)
    assert port.stopped.wait(.5)
    stream.stop()


def test_writer_cannot_erase_missed_heartbeat_before_watchdog_runs(monkeypatch):
    from g2_local.command_stream import CommandStream
    # Simulate watchdog not scheduled: writer must independently fail closed.
    monkeypatch.setattr(CommandStream, '_watch', lambda self: None)
    port = Port()
    stream = CommandStream(port, command_timeout=.3, send_timeout=.05)
    try:
        stream.wait_sent(stream.submit((1,)), timeout=.2)
        with stream.condition:
            stream.last_send = time.monotonic()-.1
            count = len(port.sent)
        assert port.stopped.wait(.6)
        assert len(port.sent) == count
        with pytest.raises(RuntimeError, match='heartbeat'):
            stream.check()
    finally:
        stream.stop()
