"""Single-writer target publisher with an independent, fail-closed watchdog.

Python scheduling is not hard real-time. An in-flight SDK call cannot be
cancelled: stop reports that explicitly rather than releasing live resources.
"""
from copy import deepcopy
import math
import threading
import time


class CommandStream:
    def __init__(self, port, *, command_timeout, send_timeout, stop_timeout=1., rate_hz=50.):
        if any(not math.isfinite(v) or v <= 0 for v in
               (command_timeout, send_timeout, stop_timeout, rate_hz)):
            raise ValueError('Positive finite stream timing required')
        self.period = 1./rate_hz
        if min(command_timeout, send_timeout) <= self.period:
            raise ValueError('Timeouts must exceed publication period')
        self.port = port
        self.command_timeout = command_timeout
        self.send_timeout = send_timeout
        self.stop_timeout = stop_timeout
        self.condition = threading.Condition()
        self.halt = threading.Event()
        self.fault = None
        self.stop_fault = None
        self.target = None
        self.sequence = 0
        self.ack_sequence = 0
        self.ack_time = None
        self.deadline = None
        self.in_flight = None
        self.last_send = None
        self.writer = self.watchdog = None

    def check(self):
        with self.condition:
            if self.fault is not None:
                raise RuntimeError(f'Command stream fault: {self.fault}') from self.fault
            if self.halt.is_set():
                raise RuntimeError('Command stream stopped; no automatic restart')

    def _fail(self, error):
        with self.condition:
            if self.fault is None:
                self.fault = error
            self.halt.set()
            self.condition.notify_all()

    def submit(self, target):
        with self.condition:
            self.check()
            if self.sequence > self.ack_sequence:
                raise RuntimeError('Previous target has not been acknowledged')
            now = time.monotonic()
            if self.deadline is not None and now >= self.deadline:
                self._fail(TimeoutError('target lease expired'))
                self.check()
            self.target = deepcopy(target)
            self.sequence += 1
            self.deadline = now + self.command_timeout
            if self.writer is None:
                self.last_send = now
                self.writer = threading.Thread(target=self._write_loop, name='g2-writer', daemon=True)
                self.watchdog = threading.Thread(target=self._watch, name='g2-watchdog', daemon=True)
                self.writer.start()
                self.watchdog.start()
            self.condition.notify_all()
            return self.sequence

    def wait_sent(self, sequence, *, timeout):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Positive finite acknowledgement timeout required')
        deadline = time.monotonic()+timeout
        with self.condition:
            if type(sequence) is not int or not 1 <= sequence <= self.sequence:
                raise ValueError('Unknown command sequence')
            while True:
                self.check()
                if self.ack_sequence == sequence:
                    return self.ack_time
                if self.ack_sequence > sequence:
                    raise RuntimeError('Acknowledgement superseded')
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    self._fail(TimeoutError('send acknowledgement timeout'))
                    self.check()
                self.condition.wait(remaining)

    def _write_loop(self):
        try:
            while not self.halt.is_set():
                with self.condition:
                    started = time.monotonic()
                    if self.halt.is_set():
                        break
                    if started >= self.deadline:
                        raise TimeoutError('target lease expired')
                    if started-self.last_send > self.send_timeout:
                        raise TimeoutError('send heartbeat missed before publication')
                    target, sequence = self.target, self.sequence
                    self.in_flight = started
                self.port.send(target)
                finished = time.monotonic()
                with self.condition:
                    self.in_flight = None
                    if finished-started > self.send_timeout:
                        raise TimeoutError('SDK send exceeded timeout')
                    if finished >= self.deadline:
                        raise TimeoutError('target lease expired during send')
                    if self.halt.is_set():
                        break  # Late SDK return is not an acknowledgement.
                    self.last_send = finished
                    if sequence > self.ack_sequence:
                        self.ack_sequence, self.ack_time = sequence, finished
                    self.condition.notify_all()
                # No burst catch-up after a delayed call.
                self.halt.wait(max(0., self.period-(time.monotonic()-started)))
        except Exception as exc:
            self._fail(exc)
        finally:
            try:
                self.port.stop()
            except Exception as exc:
                self.stop_fault = exc
                self._fail(exc)

    def _watch(self):
        while not self.halt.wait(min(.005, self.period/2)):
            with self.condition:
                now = time.monotonic()
                if now >= self.deadline:
                    self._fail(TimeoutError('target lease expired'))
                elif now-(self.in_flight if self.in_flight is not None else self.last_send) > self.send_timeout:
                    self._fail(TimeoutError('send heartbeat exceeded timeout'))

    def stop(self):
        with self.condition:
            self.halt.set()
            self.condition.notify_all()
        deadline = time.monotonic()+self.stop_timeout
        for thread in (self.writer, self.watchdog):
            if thread is not None:
                thread.join(max(0., deadline-time.monotonic()))
        if any(t is not None and t.is_alive() for t in (self.writer, self.watchdog)):
            raise TimeoutError('SDK send/hold still in flight; physical stop unconfirmed')
        if self.stop_fault is not None:
            raise RuntimeError('Measured hold failed; physical stop unconfirmed') from self.stop_fault
