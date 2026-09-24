"""Disk-backed provenance with bounded row caching and immutable checkpoints.

Historical RGB is represented by a digest of the validated float32 tensor.
Replay retains the training pixels; the journal retains every action, outcome,
scene/identity field and observation digest without duplicating image storage.
"""

from array import array
from collections import deque
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from .contract import CAMERA_KEYS
from .real_actor import validate_transition_provenance
from .training_config import canonical_json, _open_owned_regular, _unique_pairs, _reject_constant


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def compact_record(row):
    return {**{key: row[key] for key in ('reward', 'done', 'truncated', 'complementary_info')},
            'action': row['action'].detach().cpu().tolist(),
            **{field: {key: (value.detach().cpu().tolist() if key == 'observation.state'
                             else tensor_digest(value))
                       for key, value in row[field].items()}
               for field in ('state', 'next_state')}}


def validate_record(row, run_id, config_hash):
    validate_transition_provenance({**row, 'action': torch.tensor(row['action'], dtype=torch.float32)},
                                   run_id, config_hash)
    expected = {'observation.state', *(f'observation.images.{key}' for key in CAMERA_KEYS)}
    for field in ('state', 'next_state'):
        observation = row[field]
        if type(observation) is not dict or set(observation) != expected:
            raise ValueError('Invalid provenance observation schema')
        state = torch.tensor(observation['observation.state'], dtype=torch.float32)
        if state.shape != (1, 7) or not torch.isfinite(state).all().item():
            raise ValueError('Invalid provenance state')
        for key in expected - {'observation.state'}:
            digest = observation[key]
            if (type(digest) is not str or len(digest) != 64 or
                    any(char not in '0123456789abcdef' for char in digest)):
                raise ValueError('Invalid provenance RGB digest')


class ProvenanceRecords:
    """Append-only compact journal; only offsets and a bounded cache are resident."""

    def __init__(self, capacity):
        self._stream = tempfile.TemporaryFile()
        self._offsets = array('Q', [0])
        self._digest = hashlib.sha256()
        self.cache = deque(maxlen=capacity)

    def __len__(self):
        return len(self._offsets) - 1

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index >= len(self) - len(self.cache):
            return deepcopy(self.cache[index - (len(self) - len(self.cache))])
        start, end = self._offsets[index:index + 2]
        return json.loads(os.pread(self._stream.fileno(), end - start, start))

    def append(self, row):
        data = canonical_json(row) + b'\n'
        if len(data) > 131072:
            raise ValueError('Provenance record exceeds byte bound')
        start = self._offsets[-1]
        offset = 0
        while offset < len(data):
            written = os.pwrite(self._stream.fileno(), data[offset:], start + offset)
            if written <= 0:
                raise OSError('Short provenance write')
            offset += written
        self._digest.update(data)
        self._offsets.append(start + len(data))
        # Cache a detached compact value, never the incoming tensor graph.
        self.cache.append(json.loads(data))

    def descriptor(self):
        digest = self._digest.hexdigest()
        return {'format': 'rgb-sha256-v1', 'file': f'provenance-{digest}.jsonl',
                'sha256': digest, 'size': self._offsets[-1], 'count': len(self)}

    def save_snapshot(self, directory, descriptor):
        path = Path(directory) / descriptor['file']
        # Content addressing preserves older checkpoint references. A snapshot
        # never follows a concurrently appended suffix of the live journal.
        if path.exists():
            fd, _ = _open_owned_regular(path)
            with os.fdopen(fd, 'rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256') if hasattr(hashlib, 'file_digest') else None
                if digest is None:
                    digest = hashlib.sha256()
                    for data in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(data)
            if digest.hexdigest() != descriptor['sha256']:
                raise ValueError('Existing provenance snapshot integrity mismatch')
            return
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            offset = 0
            while offset < descriptor['size']:
                data = os.pread(self._stream.fileno(), min(1024 * 1024, descriptor['size'] - offset), offset)
                if not data:
                    raise ValueError('Truncated provenance journal')
                stream.write(data)
                offset += len(data)
            stream.flush()
            os.fsync(stream.fileno())

    @classmethod
    def restore(cls, directory, descriptor, *, capacity, run_id, config_hash):
        if (type(descriptor) is not dict or set(descriptor) != {'format', 'file', 'sha256', 'size', 'count'} or
                descriptor['format'] != 'rgb-sha256-v1' or
                type(descriptor['sha256']) is not str or len(descriptor['sha256']) != 64 or
                any(char not in '0123456789abcdef' for char in descriptor['sha256']) or
                descriptor['file'] != f"provenance-{descriptor['sha256']}.jsonl" or
                type(descriptor['size']) is not int or descriptor['size'] < 0 or
                type(descriptor['count']) is not int or descriptor['count'] < 0):
            raise ValueError('Invalid provenance checkpoint descriptor')
        fd, metadata = _open_owned_regular(Path(directory) / descriptor['file'])
        if metadata.st_size != descriptor['size'] or metadata.st_mode & 0o022 or metadata.st_nlink != 1:
            os.close(fd)
            raise ValueError('Invalid provenance snapshot size or ownership')
        records = cls(capacity)
        with os.fdopen(fd, 'rb') as stream:
            while data := stream.readline(131073):
                if len(data) > 131072 or not data.endswith(b'\n'):
                    raise ValueError('Invalid provenance record boundary')
                row = json.loads(data, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
                validate_record(row, run_id, config_hash)
                records.append(row)
        if records.descriptor() != descriptor:
            raise ValueError('Checkpoint provenance integrity mismatch')
        return records
