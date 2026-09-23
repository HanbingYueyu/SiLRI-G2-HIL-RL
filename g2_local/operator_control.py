"""Bounded local operator inputs for real episode control."""

import json
import os
from pathlib import Path
import stat

from .contract import EpisodeContext


class TerminalKeyReader:
    def __init__(self, source):
        self.source = source

    def poll(self):
        labels = {k.lower() for k in self.source.read_available(limit=32)
                  if isinstance(k, str) and k.lower() in ('y', 'f')}
        if labels == {'y'}:
            return 'success'
        if labels == {'f'}:
            return 'failure'
        if labels:
            raise RuntimeError('Conflicting Y/F terminal input')
        return None

    def drain(self):
        for _ in range(32):
            if not self.source.read_available(limit=32):
                return
        raise RuntimeError('Terminal input buffer exceeds drain limit')


class StartChord:
    def __init__(self, *, left_button, right_button):
        if (type(left_button) is not int or type(right_button) is not int or
                left_button not in (0, 1) or right_button not in (0, 1) or
                left_button == right_button):
            raise ValueError('Two distinct button indices required')
        self.left_button = left_button
        self.right_button = right_button
        self.reset()

    def reset(self):
        self._previous = (False, False)
        self._armed = False
        self._released = False

    def update(self, frame):
        if frame is None or getattr(frame, 'ready', True) is not True:
            self.reset()
            return False
        buttons = (frame.buttons[self.left_button], frame.buttons[self.right_button])
        if any(type(value) is not bool for value in buttons):
            self.reset()
            raise ValueError('Boolean button reports required')
        pressed = getattr(frame, 'pressed', None)
        if pressed is None:
            fresh = tuple(i for i, (old, new) in enumerate(zip(self._previous, buttons))
                          if not old and new)
        else:
            fresh = tuple((self.left_button, self.right_button).index(i)
                          for i in pressed if i in (self.left_button, self.right_button))
        result = False
        if self._armed:
            if buttons == (False, False):
                self._armed = False
                result = True
        elif buttons == (True, True) and set(fresh) == {0, 1} and not self._released:
            self._armed = True
        if result:
            self._released = True
        elif buttons == (False, False):
            self._released = False
        self._previous = buttons
        return result


class EpisodeContextInbox:
    def __init__(self, path):
        self.path = Path(path)
        self._seen = set()

    def _read_owned_regular_json(self, *, max_bytes):
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError as exc:
            raise ValueError('Context must be an owned regular file') from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise ValueError('Context must be an owned regular file')
            if info.st_size > max_bytes:
                raise ValueError('Context file too large')
            with os.fdopen(fd, 'rb', closefd=False) as source:
                data = source.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError('Context file too large')
            return json.loads(data)
        finally:
            os.close(fd)

    def read_new(self):
        payload = self._read_owned_regular_json(max_bytes=16_384)
        context = EpisodeContext.from_payload(payload)
        if context.episode_id in self._seen:
            raise ValueError('duplicate episode context')
        self._seen.add(context.episode_id)
        return context
