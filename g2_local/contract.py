"""Hardware-independent G2 action and episode contracts.

Actions are normalized XYZ + rotation-vector proposals, with no gripper slot.
Physical scale, reference frame and hardware acknowledgement belong to the driver.
"""

from dataclasses import dataclass
import math

SCHEMA_VERSION = 1
CAMERA_KEYS = ('left_wrist', 'right_aux')


def vector(values, size):
    result = tuple(float(value) for value in values)
    if len(result) != size or not all(math.isfinite(value) for value in result):
        raise ValueError(f'Expected {size} finite values')
    return result


@dataclass(frozen=True)
class EpisodeContext:
    episode_id: str
    target_offset_m: tuple[float, float, float]
    approach_source: str
    grasp_description: str

    def __post_init__(self):
        if not self.episode_id or not self.approach_source or not self.grasp_description:
            raise ValueError('Episode ID, approach and grasp metadata are required')
        object.__setattr__(self, 'target_offset_m', vector(self.target_offset_m, 3))


@dataclass(frozen=True)
class ActionDecision:
    policy_proposal: tuple[float, ...]
    human_proposal: tuple[float, ...] | None
    selected_action: tuple[float, ...]
    is_intervention: bool


def select_action(policy, *, human_active=False, human=None):
    """Arbitrate control, including explicit human zero-motion holding.

selected_action is the candidate after normalized clipping; it must not be
stored as an executed transition until the driver confirms the applied action.
"""
    proposal = vector(policy, 6)
    human_proposal = None if human is None else vector(human, 6)
    if human_active and human_proposal is None:
        raise ValueError('Active intervention requires fresh human input')
    selected = human_proposal if human_active else proposal
    clipped = tuple(max(-1., min(1., value)) for value in selected)
    return ActionDecision(proposal, human_proposal, clipped, bool(human_active))


def transition(obs, next_obs, decision, acknowledged_action, *, reward,
               terminated, truncated):
    """Build a training row only after a successful command and valid successor.

The caller validates observation freshness and command acknowledgement before
calling. acknowledged_action must be the driver's effective normalized input,
including any physical safety clipping, not measured displacement.
"""
    if obs is None or next_obs is None:
        raise ValueError('A valid predecessor and successor are required')
    action = vector(acknowledged_action, 6)
    if any(abs(value) > 1 for value in action) or not math.isfinite(float(reward)):
        raise ValueError('Invalid executed action or reward')
    return {
        'state': obs, 'next_state': next_obs, 'action': action,
        'reward': float(reward), 'done': bool(terminated),
        'truncated': bool(truncated),
        'complementary_info': {
            'schema_version': SCHEMA_VERSION,
            'is_intervention': decision.is_intervention,
            'source': 'human' if decision.is_intervention else 'policy',
            'policy_proposal': decision.policy_proposal,
            'human_proposal': decision.human_proposal,
        },
    }
