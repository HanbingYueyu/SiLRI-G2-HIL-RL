"""Read-only timing probe for a trusted local SiLRI checkpoint.

Checkpoint files MUST come from the trusted local training workflow. PyTorch
pickle loading can execute code; schema checks after loading are not a security
boundary for untrusted files. No action from this module is executable output.
"""
from copy import deepcopy
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import monotonic_ns
from traceback import clear_frames

import draccus
import numpy as np
import torch
import torch.nn.functional as F

from .contract import CAMERA_KEYS
from .policy import create_policy, create_policy_config

# Validated GDK full-frame geometry for this timing audit, not a final task ROI.
AUDIT_RGB_SHAPE = (1056, 1280, 3)


@dataclass(frozen=True)
class InferenceResult:
    cpu_prepare_ns: int
    h2d_resize_ns: int
    forward_ns: int
    total_ns: int
    cuda_allocated_bytes: int
    cuda_reserved_bytes: int
    cuda_peak_allocated_bytes: int
    action_shape: tuple[int, int]
    action_min: float
    action_max: float
    action_discarded: bool = True


def _exact_equal(actual, expected):
    """Compare serialized config recursively without bool/int coercion."""
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return actual.keys() == expected.keys() and all(
            _exact_equal(actual[key], value) for key, value in expected.items())
    if type(expected) in (list, tuple):
        return len(actual) == len(expected) and all(
            _exact_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _release_failure_frames(error):
    """Drop completed-frame locals, preserving traceback locations and chaining."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        clear_frames(current.__traceback__)
        pending.extend(link for link in (current.__cause__, current.__context__)
                       if link is not None)


class ActorInferenceProbe:
    def __init__(self, checkpoint: Path, *, device: str, warmup_steps: int):
        self.policy = None
        self._closed = False
        self._cuda_acquired = False
        self.device = device
        self.warmup_steps = warmup_steps
        self.warmup_completed = 0
        self._warmup_started = False
        try:
            self._initialize(checkpoint, device, warmup_steps)
        except BaseException as error:
            _release_failure_frames(error)
            self.close()
            raise

    def _initialize(self, checkpoint, device, warmup_steps):
        # This separate frame is complete before failure cleanup clears its tensors.
        if type(device) is not str or device != 'cuda':
            raise ValueError('device must be explicitly cuda')
        if type(warmup_steps) is not int or warmup_steps < 0:
            raise ValueError('warmup_steps must be a nonnegative integer')
        if not isinstance(checkpoint, Path):
            raise ValueError('checkpoint must be a trusted-local Path')
        self.checkpoint = checkpoint
        # Hash and deserialize the same opened file, never a second path lookup.
        with checkpoint.open('rb') as stream:
            digest = sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
            stream.seek(0)
            snapshot = torch.load(stream, map_location='cpu', weights_only=False)
        if type(snapshot) is not dict or type(snapshot.get('schema')) is not int \
                or snapshot['schema'] != 1:
            raise ValueError('checkpoint schema must be integer 1')
        if type(snapshot.get('version')) is not int or snapshot['version'] < 0:
            raise ValueError('checkpoint version must be a nonnegative integer')
        expected_config = draccus.encode(create_policy_config('cuda'))
        if not _exact_equal(snapshot.get('config'), expected_config):
            raise ValueError('checkpoint config mismatch: type, camera, state, action or options')
        weights = snapshot.get('policy')
        if not isinstance(weights, dict) or any(type(key) is not str for key in weights):
            raise ValueError('actor_keys require a string-keyed policy state dictionary')
        actor_weights = {key.removeprefix('actor.'): value for key, value in weights.items()
                         if key.startswith('actor.')}
        if not torch.cuda.is_available():
            raise ValueError('gpu CUDA is unavailable')
        gpu_name = torch.cuda.get_device_name(device)
        if type(gpu_name) is not str or gpu_name != 'NVIDIA GeForce RTX 3090':
            raise ValueError('gpu must be NVIDIA GeForce RTX 3090')
        self._cuda_acquired = True
        self.policy = create_policy(device)
        self.policy.eval()
        expected_weights = self.policy.actor.state_dict()
        if actor_weights.keys() != expected_weights.keys():
            raise ValueError('actor_keys missing or unexpected')
        for key, value in actor_weights.items():
            target = expected_weights[key]
            if type(value) is not torch.Tensor or value.shape != target.shape \
                    or value.dtype != target.dtype or value.layout != torch.strided \
                    or not torch.isfinite(value).all().item():
                raise ValueError(f'actor parameter invalid: {key}')
        self.policy.actor.load_state_dict(actor_weights, strict=True)
        self._metadata = dict(checkpoint_schema=1, checkpoint_version=snapshot['version'],
                              checkpoint_sha256=digest.hexdigest(), policy_config=expected_config,
                              gpu_name=gpu_name, device=device, warmup_steps=warmup_steps)

    def metadata(self) -> dict:
        return deepcopy(dict(self._metadata, warmup_completed=self.warmup_completed))

    def warmup(self, obs):
        """Run the requested warm-ups once; retain only their completion count."""
        try:
            if self._closed:
                raise ValueError('probe is closed')
            if self._warmup_started:
                raise ValueError('warmup already started')
            self._warmup_started = True
            for _ in range(self.warmup_steps):
                self.infer(obs)
                self.warmup_completed += 1
        except BaseException:
            self.close()
            raise

    def infer(self, obs) -> InferenceResult:
        """Measure the complete path and return scalar diagnostics only."""
        try:
            return self._infer(obs)
        except BaseException as error:
            _release_failure_frames(error)
            self.close()
            raise

    def _infer(self, obs) -> InferenceResult:
        # Keep tensor-owning work in a frame that has unwound before close().
        if self._closed:
            raise ValueError('probe is closed')
        start = monotonic_ns()
        if not isinstance(obs, Mapping):
            raise ValueError('observation must be a mapping')
        state = obs.get('state')
        if type(state) is not np.ndarray or state.dtype != np.float32 \
                or state.shape != (7,) or not state.flags.c_contiguous \
                or not np.isfinite(state).all():
            raise ValueError('observation state must be finite contiguous float32 (7,)')
        for key in CAMERA_KEYS:
            frame = obs.get(key)
            if type(frame) is not np.ndarray or frame.dtype != np.uint8 \
                    or frame.shape != AUDIT_RGB_SHAPE or not frame.flags.c_contiguous:
                raise ValueError(f'observation {key} must be contiguous uint8 '
                                 f'{AUDIT_RGB_SHAPE} RGB for the full-frame timing audit')
        # Clear outstanding CUDA work before timing this observation's stages.
        torch.cuda.synchronize(self.device)
        cpu_start = monotonic_ns()
        batch = {'observation.state': torch.from_numpy(state).unsqueeze(0)}
        for key in CAMERA_KEYS:
            batch[f'observation.images.{key}'] = (
                torch.from_numpy(obs[key]).permute(2, 0, 1).unsqueeze(0)
                .to(dtype=torch.float32).div_(255.0))
        cpu_end = monotonic_ns()
        with torch.inference_mode():
            for key, tensor in batch.items():
                batch[key] = tensor.to(self.device)
            for key in CAMERA_KEYS:
                name = f'observation.images.{key}'
                batch[name] = F.interpolate(batch[name], size=(128, 128), mode='bilinear',
                                            align_corners=False, antialias=False)
            torch.cuda.synchronize(self.device)
            resize_end = monotonic_ns()
            output = self.policy.select_action(batch)
            torch.cuda.synchronize(self.device)
            forward_end = monotonic_ns()
            if type(output) is not tuple or len(output) != 2:
                raise ValueError('action result must be the SiLRI (tensor, info) tuple')
            action, _ = output
            if type(action) is not torch.Tensor or action.shape != (1, 6) \
                    or action.dtype != torch.float32 or not action.is_cuda:
                raise ValueError('action must be a CUDA float32 tensor of shape (1, 6)')
            # Only these six diagnostic values leave CUDA; never return an action.
            diagnostic = action.detach().cpu()
            if not torch.isfinite(diagnostic).all().item() \
                    or (diagnostic < -1 + 1e-6).any().item() \
                    or (diagnostic > 1 - 1e-6).any().item():
                raise ValueError('action must be finite and inside the SiLRI clamp')
            action_min = float(diagnostic.min().item())
            action_max = float(diagnostic.max().item())
            del diagnostic, action, output, batch
        memory = [torch.cuda.memory_allocated(self.device),
                  torch.cuda.memory_reserved(self.device),
                  torch.cuda.max_memory_allocated(self.device)]
        if any(type(value) is not int or value < 0 for value in memory):
            raise ValueError('CUDA memory statistics must be nonnegative integers')
        return InferenceResult(cpu_prepare_ns=cpu_end - cpu_start,
                               h2d_resize_ns=resize_end - cpu_end,
                               forward_ns=forward_end - resize_end,
                               total_ns=monotonic_ns() - start,
                               cuda_allocated_bytes=memory[0],
                               cuda_reserved_bytes=memory[1],
                               cuda_peak_allocated_bytes=memory[2],
                               action_shape=(1, 6), action_min=action_min,
                               action_max=action_max)

    def close(self):
        """Release owned model references and the CUDA allocator cache once."""
        if self._closed:
            return
        self._closed = True
        self.policy = None
        if self._cuda_acquired:
            try:
                torch.cuda.empty_cache()
            except BaseException as error:
                # A cleanup interrupt must not resemble an ordinary clean Ctrl+C.
                raise RuntimeError(f'Actor CUDA cleanup failed: {type(error).__name__}: {error}') from error
