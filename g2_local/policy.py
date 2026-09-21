"""Explicit dual-camera SiLRI configuration for software integration checks."""
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.silri.configuration_silri import SiLRIConfig
from lerobot.policies.silri.modeling_silri import SiLRIPolicy
from .contract import CAMERA_KEYS


def create_policy(device='cpu'):
    features = {'observation.state': PolicyFeature(FeatureType.STATE, (7,))}
    features.update({f'observation.images.{key}': PolicyFeature(FeatureType.VISUAL, (3, 128, 128))
                     for key in CAMERA_KEYS})
    config = SiLRIConfig(input_features=features,
                        output_features={'action': PolicyFeature(FeatureType.ACTION, (6,))},
                        device=device, use_torch_compile=False, shared_encoder=False,
                        normalization_mapping={}, dataset_stats={},
                        freeze_vision_encoder=False, latent_dim=32)
    # This uses the upstream small CNN, trained from scratch. It is not the
    # pretrained ResNet recipe and makes no task-learning performance claim.
    return SiLRIPolicy(config).to(device)
