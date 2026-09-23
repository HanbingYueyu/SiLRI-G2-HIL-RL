"""Explicit local task configuration, with no inherited robot motion limits."""
from dataclasses import dataclass
import math
from .contract import CAMERA_KEYS, EpisodeContext, vector


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


@dataclass(frozen=True)
class HingeInsertTaskConfig:
    """MVP algorithm defaults for the final local hinge-insertion phase.

    These are policy/reward starting points from the deployment guide, not
    robot safety limits. Absolute workspace bounds remain mandatory and must be
    supplied by the operator before constructing a motion configuration.
    """
    control_hz: float = 10.
    max_episode_steps: int = 80
    fix_gripper: bool = True
    action_scale: tuple[float, ...] = (.0015, .0015, .0015, .026, .026, .026)
    success_reward: float = 10.
    failure_reward: float = -1.
    step_reward: float = -.05
    reward_source: str = 'human'
    target_xy_range_m: float = 0.
    ee_xyz_range_m: float = .003
    ee_rpy_range_rad: float = .008726646259971648

    def __post_init__(self):
        if not isinstance(self.fix_gripper, bool) or not self.fix_gripper:
            raise ValueError('Hinge MVP requires a fixed closed gripper')
        if type(self.control_hz) not in (int, float) or not math.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError('control_hz must be positive and finite')
        if type(self.max_episode_steps) is not int or self.max_episode_steps <= 0:
            raise ValueError('max_episode_steps must be a positive integer')
        object.__setattr__(self, 'action_scale', vector(self.action_scale, 6))
        if any(value <= 0 for value in self.action_scale):
            raise ValueError('Hinge action scales must be positive')
        for name in ('success_reward', 'failure_reward', 'step_reward', 'target_xy_range_m',
                     'ee_xyz_range_m', 'ee_rpy_range_rad'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
        if self.target_xy_range_m < 0 or self.ee_xyz_range_m < 0 or self.ee_rpy_range_rad < 0:
            raise ValueError('Randomization ranges must be nonnegative')
        if self.reward_source not in ('human', 'classifier'):
            raise ValueError('reward_source must be human or classifier')

    def motion_config(self, *, workspace_low=None, workspace_high=None):
        """Build the low-level config without silently inventing workspace bounds."""
        return LocalTaskConfig(action_scale=self.action_scale,
                               workspace_low=workspace_low,
                               workspace_high=workspace_high)

    def sample_episode_context(self, rng, *, episode_id, approach_source,
                               grasp_description):
        """Sample explicit domain-randomization metadata; never reset hardware.

        ``target_xy_range_m`` describes the target/fridge perturbation while
        ``ee_xyz_range_m`` and ``ee_rpy_range_rad`` describe the end-effector
        reset perturbation.  Keeping the two tuples separate makes the source
        of each variation visible in replay and evaluation records.
        """
        uniform = getattr(rng, 'uniform', None)
        if not callable(uniform):
            raise ValueError('A random generator with uniform() is required')

        def draw(limit):
            value = uniform(-limit, limit)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('Random generator returned a non-finite value')
            return float(value)

        target_offset = (draw(self.target_xy_range_m),
                         draw(self.target_xy_range_m), 0.)
        ee_reset = tuple(draw(self.ee_xyz_range_m) for _ in range(3)) + tuple(
            draw(self.ee_rpy_range_rad) for _ in range(3))
        return EpisodeContext(episode_id, target_offset, approach_source,
                              grasp_description, ee_reset)
