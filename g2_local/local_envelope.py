"""Per-episode base-frame offsets; no hardware access or motion authorization."""
from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial.transform import Rotation
from .contract import vector


@dataclass(frozen=True)
class LocalEnvelope:
    translation_low_m: tuple
    translation_high_m: tuple
    rotation_max_rad: float

    def __post_init__(self):
        for key in ('translation_low_m', 'translation_high_m'):
            object.__setattr__(self, key, vector(getattr(self, key), 3))
        if any(not low < 0 < high for low, high in
               zip(self.translation_low_m, self.translation_high_m)):
            raise ValueError('Local translation bounds must strictly contain the origin')
        if (type(self.rotation_max_rad) not in (int, float) or
                not math.isfinite(self.rotation_max_rad) or
                not 0 < self.rotation_max_rad <= math.pi):
            raise ValueError('Local rotation bound must be in (0, pi] radians')

    def check(self, pose, reference):
        pose, reference = (np.asarray(vector(value, 7)) for value in (pose, reference))
        for value in (pose, reference):
            if abs(np.linalg.norm(value[3:]) - 1.) > .01:
                raise ValueError('Local envelope requires unit quaternions')
        delta = pose[:3] - reference[:3]
        if (np.any(delta < np.asarray(self.translation_low_m)) or
                np.any(delta > np.asarray(self.translation_high_m))):
            raise ValueError('Pose outside episode local translation envelope')
        relative = Rotation.from_quat(pose[3:]) * Rotation.from_quat(reference[3:]).inv()
        if relative.magnitude() > self.rotation_max_rad:
            raise ValueError('Pose outside episode local rotation envelope')
