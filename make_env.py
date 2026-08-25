import traceback
import sys
import copy

def print_green(x):
    return print("\033[92m {}\033[00m".format(x))



def make_env(env_cfg, fake_env, use_human_intervention, classifier=False, use_gripper_penalty=False, cfg=None):
    try:  
        if env_cfg.robot_config.robot_type == "sim":
            import gymnasium as gym
            import gym_hil  
            from omegaconf import OmegaConf
            from lerobot.configs.types import FeatureType, PolicyFeature
            import copy

            def build_expert_env_cfg(cfg, env_cfg):
                """Env config for ExpertControlWrapper.make_policy."""
                features = {}
                features_map = {}
                for cam in env_cfg.lerobot.expert_cameras:
                    key = f"observation.images.{cam}"
                    features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(128, 128, 3))
                    features_map[key] = key
                features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(env_cfg.lerobot.expert_state_dim,))
                features_map["observation.state"] = "observation.state"
                if hasattr(env_cfg.lerobot, "expert_env_state_dim") :
                    features["observation.environment_state"] = PolicyFeature(
                        type=FeatureType.ENVIRONMENT_STATE, shape=(env_cfg.lerobot.expert_env_state_dim,)
                    )
                    features_map["observation.environment_state"] = "observation.environment_state"
                features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=(env_cfg.lerobot.action_dim,))
                features_map["action"] = "action" 

                expert_env = copy.deepcopy(cfg.env)
                expert_env.features = features
                expert_env.features_map = features_map
                return expert_env


            cfg.expert_policy.device = env_cfg.device
            cfg.expert_policy.storage_device = env_cfg.device
            cfg.expert_policy.num_discrete_actions = cfg.policy.num_discrete_actions
            expert_cfg = {
                "expert_model_cfg": cfg.expert_policy,
                "env_cfg": build_expert_env_cfg(cfg, env_cfg),
                "expert_cfg": env_cfg.expert,
                "device": env_cfg.device,
            }

            env_cfg = OmegaConf.to_container(env_cfg.env_cfg, resolve=True)
            env_cfg["expert_cfg"] = expert_cfg
            env = gym.make(**env_cfg)
        else:
            from rl_envs.base_env import BaseEnv
            from rl_envs.wrappers import HumanIntervention, SERLObsWrapper, AugmentedObservationWrapper
            from rl_envs.reward_wrapper import MultiCameraBinaryRewardClassifierWrapper, GripperPenaltyWrapper

            env = BaseEnv(config=env_cfg.robot_config, fake_env=fake_env)
            
            if not fake_env and use_human_intervention:
                intervention_backend = getattr(env_cfg, "intervention_backend", "xtele")
                if intervention_backend == "spacemouse":
                    assert env_cfg.robot_config.dual_arm == False, "spacemouse intervention is not supported for dual arm robots"
                    spacemouse_enable_gripper = bool(getattr(env_cfg, "spacemouse_enable_gripper", True))
                    if getattr(env_cfg.robot_config, "fix_gripper", False):
                        # Keep action format consistent, but disable manual gripper toggles when gripper is fixed.
                        spacemouse_enable_gripper = False
                    from rl_envs.wrappers import SpaceMouseIntervention
                    env = SpaceMouseIntervention(
                        env,
                        deadzone=getattr(env_cfg, "spacemouse_deadzone", 1e-3),
                        axis_deadzone=getattr(env_cfg, "spacemouse_axis_deadzone", None),
                        enable_gripper=spacemouse_enable_gripper,
                        translation_scale=getattr(env_cfg, "spacemouse_translation_scale", 1.0),
                        rotation_scale=getattr(env_cfg, "spacemouse_rotation_scale", 1.0),
                        axis_signs=getattr(env_cfg, "spacemouse_axis_signs", [1, 1, 1, 1, 1, 1]),
                    )
                elif intervention_backend == "xtele":
                    env = HumanIntervention(env)
                else:
                    raise ValueError(f"Unsupported intervention backend: {intervention_backend}")
            
            env = AugmentedObservationWrapper(env)
            env = SERLObsWrapper(env,proprio_keys=env_cfg.robot_config.proprio_keys, use_force=env_cfg.use_force)
            if classifier:
                env = MultiCameraBinaryRewardClassifierWrapper(env, env_cfg.robot_config.classifier_cfg, cfg=cfg)
                if use_gripper_penalty:
                    env = GripperPenaltyWrapper(env, penalty=env_cfg.robot_config.gripper_penalty)
    except Exception as e:
        print_green(f"[{type(e).__name__}] {e!r}")
        traceback.print_exc()          # full stacktrace
        sys.exit(1)
    return env