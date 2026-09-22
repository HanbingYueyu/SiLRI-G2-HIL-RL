from pathlib import Path
from types import SimpleNamespace
import signal

from g2_local.real_actor_audit import run_session


class FakeProcess:
    def __init__(self, command, *, output=None, returncode=0):
        self.command = command
        self.returncode = None
        self._final = returncode
        self.signals = []
        if output is not None:
            output.mkdir(parents=True)
            (output / 'clock.sock').touch()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        del timeout
        if self.returncode is None:
            self.returncode = self._final
        return self.returncode

    def send_signal(self, signum):
        self.signals.append(signum)

    def terminate(self):
        self.returncode = self._final


class FakeClient:
    def __init__(self):
        self.closed = False

    def read(self):
        return SimpleNamespace(healthy=True, reason='ok')

    def close(self):
        self.closed = True


def test_run_session_waits_for_healthy_monitor_then_stops_it_after_audit(tmp_path):
    commands = []
    processes = []
    monitor_output = tmp_path / 'session' / 'monitor'
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'fixture')

    def popen(command, **kwargs):
        del kwargs
        commands.append(command)
        if any('clock_monitor' in item for item in command):
            process = FakeProcess(command, output=monitor_output)
        else:
            process = FakeProcess(command)
        processes.append(process)
        return process

    client = FakeClient()
    result = run_session(
        output=tmp_path / 'session', master='044052.fffe.000010',
        monitor_seconds=300, audit_seconds=120,
        checkpoint=checkpoint, warmup_steps=10,
        popen_factory=popen, client_factory=lambda **kwargs: client,
        sleep_fn=lambda _: None)

    assert result == 0
    assert len(commands) == 2
    assert commands[0][commands[0].index('-m') + 1] == 'g2_local.clock_monitor'
    assert commands[1][commands[1].index('-m') + 1] == 'g2_local.freshness_audit'
    assert '--device' in commands[1] and commands[1][commands[1].index('--device') + 1] == 'cuda'
    assert '--inference-delay-s' not in commands[1]
    assert processes[0].signals == [signal.SIGINT]
    assert client.closed is True


def test_run_session_does_not_start_audit_when_monitor_never_becomes_healthy(tmp_path):
    commands = []
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'fixture')

    class DeadMonitor(FakeProcess):
        def poll(self):
            return 2

    def popen(command, **kwargs):
        del kwargs
        commands.append(command)
        return DeadMonitor(command)

    try:
        run_session(
            output=tmp_path / 'session', master='044052.fffe.000010',
            monitor_seconds=300, audit_seconds=120,
            checkpoint=checkpoint, popen_factory=popen,
            client_factory=lambda **kwargs: FakeClient(), sleep_fn=lambda _: None)
    except RuntimeError as error:
        assert 'monitor' in str(error)
    else:
        raise AssertionError('expected monitor preflight failure')
    assert len(commands) == 1
