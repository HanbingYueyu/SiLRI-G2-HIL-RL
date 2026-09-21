"""Explicit local task configuration, with no inherited robot motion limits."""
from dataclasses import dataclass
from .contract import CAMERA_KEYS, vector


@dataclass(frozen=True)
class LocalTaskConfig:
    camera_keys: tuple[str, ...] = CAMERA_KEYS
    action_scale: tuple[float, ...] | None = None
    workspace_low: tuple[float, ...] | None = None
    workspace_high: tuple[float, ...] | None = None

    def __post_init__(self):
        keys = tuple(self.camera_keys)
        if len(keys) < 2 or len(set(keys)) != len(keys):
            raise ValueError('At least two distinct camera keys required')
        if not all(isinstance(key, str) and key.strip() for key in keys):
            raise ValueError('Camera keys must be nonempty strings')
        object.__setattr__(self, 'camera_keys', keys)
        for name, size in (('action_scale', 6), ('workspace_low', 3),
                           ('workspace_high', 3)):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, vector(value, size))

    def validate_motion(self):
        """Validate configured limits only; this does not authorize hardware motion.

        Scales are metres and radians per policy step. Workspace bounds are
        base_link positions. Driver permission and live state checks are separate.
        """
        if any(value is None for value in
               (self.action_scale, self.workspace_low, self.workspace_high)):
            raise ValueError('Explicit action scale and workspace required')
        if any(value <= 0 for value in self.action_scale):
            raise ValueError('Action scales must be positive')
        if any(low >= high for low, high in
               zip(self.workspace_low, self.workspace_high)):
            raise ValueError('Workspace lower bounds must be below upper bounds')
