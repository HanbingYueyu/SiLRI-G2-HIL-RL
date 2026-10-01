"""Operator-facing console lines.

Stdout is for the human at the terminals: it never feeds the control path,
the freshness guards or the evidence files. Every line is timestamped and
flushed so a waiting process never looks frozen.
"""

import sys
import time


def stamp():
    return time.strftime('%H:%M:%S')


def say(message, *, stream=None):
    stream = sys.stdout if stream is None else stream
    print(f'[{stamp()}] {message}', file=stream, flush=True)


class RateLimit:
    """Bounded, non-blocking repeat filter for periodic console lines."""

    def __init__(self, interval_s, *, monotonic=None):
        if type(interval_s) not in (int, float) or interval_s < 0:
            raise ValueError('Non-negative console interval required')
        self.interval_s = float(interval_s)
        self.monotonic = time.monotonic if monotonic is None else monotonic
        self._last = None

    def due(self):
        now = self.monotonic()
        if self._last is not None and now - self._last < self.interval_s:
            return False
        self._last = now
        return True


def number(value, *, digits=4):
    """Format a metric for a console line without raising on None/NaN/inf."""
    if value is None:
        return '—'
    try:
        value = float(value)
    except (TypeError, ValueError):
        return '—'
    if value != value or value in (float('inf'), float('-inf')):
        return str(value)
    return f'{value:.{digits}f}'
