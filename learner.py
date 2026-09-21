# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Learner server runner for distributed HILSerl robot policy training.

This script implements the learner component of the distributed HILSerl architecture.
It initializes the policy network, maintains replay buffers, and updates
the policy based on transitions received from the actor server.

Examples of usage:

- Start a learner server for training:
```bash
python -m lerobot.scripts.rl.learner --config_path src/lerobot/configs/train_config_hilserl_so100.json
```

**NOTE**: Start the learner server before launching the actor server. The learner opens a gRPC server
to communicate with actors.

**NOTE**: Training progress can be monitored through Weights & Biases if wandb.enable is set to true
in your configuration.

**WORKFLOW**:
1. Create training configuration with proper policy, dataset, and environment settings
2. Start this learner server with the configuration
3. Start an actor server with the same configuration
4. Monitor training progress through wandb dashboard

For more details on the complete HILSerl training workflow, see:
https://github.com/michel-aractingi/lerobot-hilserl-guide
"""

import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pprint import pformat

import grpc
import torch
from termcolor import colored
from torch import nn
from torch.multiprocessing import Queue
from torch.optim.optimizer import Optimizer

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
)
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_policy_config
# from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.robots import so100_follower  # noqa: F401
from lerobot.scripts.rl import learner_service
from lerobot.teleoperators import gamepad, so101_leader  # noqa: F401
from lerobot.transport import services_pb2_grpc
from lerobot.transport.utils import (
    MAX_MESSAGE_SIZE,
    bytes_to_python_object,
    bytes_to_transitions,
    state_to_bytes,
)
from lerobot.utils.buffer import ReplayBuffer, concatenate_batch_transitions
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state as utils_load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.transition import move_state_dict_to_device, move_transition_to_device
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs.types import FeatureType, PolicyFeature

from lerobot.utils.wandb_utils import WandBLogger
import hydra
from hydra.core.hydra_config import HydraConfig
import draccus
from omegaconf import OmegaConf
from lerobot.configs.default import DatasetConfig
import cv2
import numpy as np
from safetensors.torch import load_file
LOG_PREFIX = "[LEARNER]"
expert_training_step = 0

new_offline_transition_num = 0

#################################################
# MAIN ENTRY POINTS AND CORE ALGORITHM FUNCTIONS #
#################################################


# @parser.wrap()
@hydra.main(config_path="./cfg", config_name="config", version_base=None) 
def train_cli(env_cfg):
    cfg_dir = Path(__file__).resolve().parent / "cfg"
    if env_cfg.robot_config.robot_type == "ur_wrist":
        env_cfg.lerobot_config_path = str(cfg_dir / "train_config_silri_ur.json")
    elif "franka" in env_cfg.robot_config.robot_type:
        env_cfg.lerobot_config_path = str(cfg_dir / "train_config_silri_franka.json")
    elif "tienkung" in env_cfg.robot_config.robot_type:
        env_cfg.lerobot_config_path = str(cfg_dir / "train_config_silri_tienkung.json")
    elif "sim" in env_cfg.robot_config.robot_type:
        env_cfg.lerobot_config_path = str(cfg_dir / "train_config_simulator.json")
    else:
        raise ValueError(f"Invalid robot type: {env_cfg.robot_type}")

    config_path = env_cfg.lerobot_config_path

    # Load the config directly with draccus.parse, passing overrides through args
    with draccus.config_type("json"):
        if not env_cfg.fix_gripper:
            cfg = draccus.parse(TrainRLServerPipelineConfig, config_path, args=[f"--policy.type={env_cfg.policy_type}",f"--policy.num_discrete_actions=2", f"--policy.actor_learner_config.learner_port={env_cfg.learner_port}", f"--seed={env_cfg.seed}"])
        else:
            cfg = draccus.parse(TrainRLServerPipelineConfig, config_path, args=[f"--policy.type={env_cfg.policy_type}", f"--policy.actor_learner_config.learner_port={env_cfg.learner_port}", f"--seed={env_cfg.seed}"])
    
    if "sim" in env_cfg.robot_config.robot_type:
        apply_sim_task_overrides(cfg, env_cfg)

    # Safely override dataset only if provided in env_cfg, converting Hydra DictConfig to DatasetConfig
    if hasattr(env_cfg, "dataset") and env_cfg.dataset is not None:
        try:
            dataset_obj = OmegaConf.to_object(env_cfg.dataset)
            if isinstance(dataset_obj, dict):
                cfg.dataset = DatasetConfig(**dataset_obj)
            else:
                cfg.dataset = dataset_obj
            
        except Exception as e:
            print(f"WARN: Ignoring invalid dataset override: {e}")
            exit(-1)


    if env_cfg.resume_model:
        cfg.resume = True
        cfg.output_dir = Path(env_cfg.resume_path)
    else:
        cfg.resume = False
        # Keep as str so validate() does not raise FileExistsError on Hydra's already-created run dir.
        # Example: experiments/<task>/exp_local/<date>/<overrides>/<seed>
        cfg.output_dir = os.path.abspath(HydraConfig.get().runtime.output_dir)
    cfg.job_name = env_cfg.task_name
    cfg.validate(config_path)
    cfg.wandb.name = env_cfg.task_name
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")
    # Use the job_name from the config
    
    train(
        cfg,
        job_name=env_cfg.task_name,
        env_cfg=env_cfg,
    )

    logging.info("[LEARNER] train_cli finished")


def train(cfg: TrainRLServerPipelineConfig, job_name: str | None = None, env_cfg: any = None):
    """
    Main training function that initializes and runs the training process.

    Args:
        cfg (TrainRLServerPipelineConfig): The training configuration
        job_name (str | None, optional): Job name for logging. Defaults to None.
    """

    # cfg.validate()

    if job_name is None:
        raise ValueError("Job name must be specified either in config or as a parameter")

    display_pid = False
    if not use_threads(cfg):
        display_pid = True

    # Create logs directory to ensure it exists
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"learner_{job_name}.log")

    # Initialize logging with explicit log file
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Learner logging initialized, writing to {log_file}")
    logging.info(pformat(cfg.to_dict()))

    try:
        # Setup WandB logging if enabled
        if cfg.wandb.enable and cfg.wandb.project:
            from lerobot.utils.wandb_utils import WandBLogger
            wandb_logger = WandBLogger(cfg)
        else:
            wandb_logger = None
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    except Exception as e:
        traceback.print_exc()
        exit(-1)

    # print("successfully create wandb logger...")
    # input("Press Enter to continue...")
    # Handle resume logic
    cfg = handle_resume_logic(cfg)

    set_seed(seed=cfg.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event
    start_learner_threads(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
        env_cfg=env_cfg,
    )


def start_learner_threads(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,  # Event,
    env_cfg: any,
) -> None:
    """
    Start the learner threads for training.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        wandb_logger (WandBLogger | None): Logger for metrics
        shutdown_event: Event to signal shutdown
    """
    # Create multiprocessing queues
    transition_queue = Queue()
    interaction_message_queue = Queue()
    parameters_queue = Queue()

    concurrency_entity = None

    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from torch.multiprocessing import Process

        concurrency_entity = Process

    communication_process = concurrency_entity(
        target=start_learner,
        args=(
            parameters_queue,
            transition_queue,
            interaction_message_queue,
            shutdown_event,
            cfg,
        ),
        daemon=True,
    )
    communication_process.start()

    add_actor_information_and_train(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        parameters_queue=parameters_queue,
        env_cfg=env_cfg,
    )
    logging.info("[LEARNER] Training process stopped")

    logging.info("[LEARNER] Closing queues")
    transition_queue.close()
    interaction_message_queue.close()
    parameters_queue.close()

    communication_process.join()
    logging.info("[LEARNER] Communication process joined")

    logging.info("[LEARNER] join queues")
    transition_queue.cancel_join_thread()
    interaction_message_queue.cancel_join_thread()
    parameters_queue.cancel_join_thread()

    logging.info("[LEARNER] queues closed")


#######################################compute_target_prob##########
# Core algorithm functions #
#################################################


def add_actor_information_and_train(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,  # Event,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    parameters_queue: Queue,
    env_cfg: any,
):
    """
    Handles data transfer from the actor to the learner, manages training updates,
    and logs training progress in an online reinforcement learning setup.

    This function continuously:
    - Transfers transitions from the actor to the replay buffer.
    - Logs received interaction messages.
    - Ensures training begins only when the replay buffer has a sufficient number of transitions.
    - Samples batches from the replay buffer and performs multiple critic updates.
    - Periodically updates the actor, critic, and temperature optimizers.
    - Logs training statistics, including loss values and optimization frequency.

    NOTE: This function doesn't have a single responsibility, it should be split into multiple functions
    in the future. The reason why we did that is the  GIL in Python. It's super slow the performance
    are divided by 200. So we need to have a single thread that does all the work.

    Args:
        cfg (TrainRLServerPipelineConfig): Configuration object containing hyperparameters.
        wandb_logger (WandBLogger | None): Logger for tracking training progress.
        shutdown_event (Event): Event to signal shutdown.
        transition_queue (Queue): Queue for receiving transitions from the actor.
        interaction_message_queue (Queue): Queue for receiving interaction messages from the actor.
        parameters_queue (Queue): Queue for sending policy parameters to the actor.
    """
    # Extract all configuration variables at the beginning, it improve the speed performance
    # of 7%
    device = get_safe_torch_device(try_device=cfg.policy.device, log=True)
    storage_device = get_safe_torch_device(try_device=cfg.policy.storage_device)
    clip_grad_norm_value = cfg.policy.grad_clip_norm
    online_step_before_learning = cfg.policy.online_step_before_learning
    utd_ratio = cfg.policy.utd_ratio
    fps = cfg.env.fps
    log_freq = cfg.log_freq
    save_freq = cfg.save_freq
    policy_update_freq = cfg.policy.policy_update_freq
    policy_parameters_push_frequency = cfg.policy.actor_learner_config.policy_parameters_push_frequency
    saving_checkpoint = cfg.save_checkpoint
    online_steps = cfg.policy.online_steps
    async_prefetch = cfg.policy.async_prefetch

    # Initialize logging for multiprocessing
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_train_process_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Initialized logging for actor information and training process")

    logging.info("Initializing policy")


    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )


    assert isinstance(policy, nn.Module)

    policy.train()

    push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)

    last_time_policy_pushed = time.time()

    optimizers, lr_scheduler = policy.get_optimizer_and_scheduler()

    # If we are resuming, we need to load the training state
    resume_optimization_step, resume_interaction_step = load_training_state(cfg=cfg, optimizers=optimizers)
    
    log_training_info(cfg=cfg, policy=policy)
    replay_buffer = initialize_replay_buffer(cfg, device, storage_device)
    batch_size = cfg.batch_size
    offline_replay_buffer = None
    try:
        offline_replay_buffer = initialize_offline_replay_buffer(
            cfg=cfg,
            device=device,
            storage_device=storage_device,
        )
    except Exception as e:
        print(f"WARN: Ignoring invalid dataset override: {e}")
        exit(0)

    """
        When only the offline_replay_buffer is used, the batch size does not need halving
    """
    if "hgdagger" not in cfg.policy.type:
    # if not cfg.policy.only_off_and_intervention:
        batch_size: int = batch_size // 2  # We will sample from both replay buffer

    logging.info("Starting learner thread")
    interaction_message = None
    optimization_step = resume_optimization_step if resume_optimization_step is not None else 0
    interaction_step_shift = resume_interaction_step if resume_interaction_step is not None else 0

    dataset_repo_id = None
    if cfg.dataset is not None:
        dataset_repo_id = cfg.dataset.repo_id

    # Initialize iterators
    online_iterator = None
    offline_iterator = None


    # =================== offline training ===================
    if "silri" in cfg.policy.type and not cfg.resume:
        offline_training(
            cfg=cfg,
            policy=policy,
            optimizers=optimizers,
            offline_replay_buffer=offline_replay_buffer,
            wandb_logger=wandb_logger,
        )
    
    # st_time = time.time()
    # NOTE: THIS IS THE MAIN LOOP OF THE LEARNER

    while True:
        # Exit the training loop if shutdown is requested
        if shutdown_event is not None and shutdown_event.is_set():
            logging.info("[LEARNER] Shutdown signal received. Exiting...")
            break

        # Process all available transitions to the replay buffer, send by the actor server
        process_transitions(
            transition_queue=transition_queue,
            replay_buffer=replay_buffer,
            offline_replay_buffer=offline_replay_buffer,
            device=device,
            dataset_repo_id=dataset_repo_id,
            shutdown_event=shutdown_event,
            optimizers=optimizers,
            policy=policy,
            clip_grad_norm_value=clip_grad_norm_value,
            batch_size=batch_size,
            async_prefetch=async_prefetch,
            wandb_logger=wandb_logger,
            cfg=cfg,
            optimization_step=optimization_step
        )


        # Process all available interaction messages sent by the actor server
        interaction_message = process_interaction_messages(
            interaction_message_queue=interaction_message_queue,
            interaction_step_shift=interaction_step_shift,
            wandb_logger=wandb_logger,
            shutdown_event=shutdown_event,
        )
        
        # Check if training is complete (message from actor)
        if interaction_message is not None and interaction_message.get("training_complete", False):
            logging.info("[LEARNER] Received training complete message from Actor. Saving final checkpoint...")
            try:
                save_training_checkpoint(
                    cfg=cfg,
                    optimization_step=optimization_step,
                    online_steps=online_steps,
                    interaction_message=interaction_message,
                    policy=policy,
                    optimizers=optimizers,
                    replay_buffer=replay_buffer,
                    offline_replay_buffer=offline_replay_buffer,
                    dataset_repo_id=dataset_repo_id,
                    fps=fps,
                )
                logging.info("[LEARNER] Final checkpoint saved successfully")
            except Exception as e:
                logging.error(f"[LEARNER] Failed to save final checkpoint: {e}")
                traceback.print_exc()
            input("Press Enter to shut down all processes...")
            # Set shutdown event to exit training loop
            if shutdown_event is not None:
                shutdown_event.set()
                logging.info("[LEARNER] Shutdown event set due to training completion")
            break

        # Wait until the replay buffer has enough samples to start training
        if len(replay_buffer) < online_step_before_learning  or len(offline_replay_buffer) < online_step_before_learning:
            continue

        if online_iterator is None:
            online_iterator = replay_buffer.get_iterator(
                batch_size=batch_size, async_prefetch=async_prefetch, queue_size=2
            )

        if offline_replay_buffer is not None and offline_iterator is None:
            offline_iterator = offline_replay_buffer.get_iterator(
                batch_size=batch_size, async_prefetch=async_prefetch, queue_size=2
            )

        time_for_one_optimization_step = time.time()
        
        # First utd_ratio - 1 updates: critic only
        for _ in range(utd_ratio - 1):
            # Sample from the iterators
            """
                Only offline data plus human-intervention data is needed
            """
            if "hgdagger" in cfg.policy.type:
            # if cfg.policy.only_off_and_intervention:
                if offline_replay_buffer is not None:
                    batch_offline = next(offline_iterator)
                    batch_offline['is_intervention'] = torch.ones_like(batch_offline['done']).to(device)
                    batch = batch_offline
            else:
                batch = next(online_iterator)
                batch['is_intervention'] = batch["complementary_info"]["is_intervention"]
            
                if offline_replay_buffer is not None:
                    batch_offline = next(offline_iterator)
                    batch_offline['is_intervention'] = torch.ones_like(batch_offline['done']).to(device)
                    batch = concatenate_batch_transitions(
                        left_batch_transitions=batch, right_batch_transition=batch_offline
                    )
            
            check_nan_in_transition(observations=batch["state"], actions=batch["action"], next_state=batch["next_state"])

            observation_features, next_observation_features = get_observation_features(
                policy=policy, observations=batch["state"], next_observations=batch["next_state"]
            )

            # Create a batch dictionary with all required elements for the forward method
            forward_batch = batch
            forward_batch['observation_feature'] = observation_features
            forward_batch['next_observation_feature'] = next_observation_features


            """
                hgdagger is pure imitation learning and needs no critic
            """
            if "critic" in optimizers.keys():
                # Use the forward method for critic loss
                critic_output = policy.forward(forward_batch, model="critic")

                # Main critic optimization
                loss_critic = critic_output["loss_critic"]
                
                optimizers["critic"].zero_grad()
                loss_critic.backward()

                if hasattr(policy, "critic_ensemble"):
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                        parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
                    )
                elif hasattr(policy, "critic_qc"):
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                        parameters=policy.critic_qc.parameters(), max_norm=clip_grad_norm_value
                    )
                optimizers["critic"].step()

            # Discrete critic optimization (if available)
            if "discrete_critic" in optimizers.keys():
                discrete_critic_output = policy.forward(forward_batch, model="discrete_critic")
                loss_discrete_critic = discrete_critic_output["loss_discrete_critic"]
                optimizers["discrete_critic"].zero_grad()
                loss_discrete_critic.backward()
                optimizers["discrete_critic"].step()
                

            # Update target networks (main and discrete)
            policy.update_target_networks()

        # Sample for the last update in the UTD ratio
        # Final (utd_ratio-th) update: critic plus actor
        """
            Only offline data plus human-intervention data is needed
        """
        if "hgdagger" in cfg.policy.type:
        # if cfg.policy.only_off_and_intervention:
            if offline_replay_buffer is not None:
                batch_offline = next(offline_iterator)
                batch_offline['is_intervention'] = torch.ones_like(batch_offline['done']).to(device)
                batch = batch_offline
        else:
            batch = next(online_iterator)
            batch['is_intervention'] = batch["complementary_info"]["is_intervention"]
        
            if offline_replay_buffer is not None:
                batch_offline = next(offline_iterator)
                batch_offline['is_intervention'] = torch.ones_like(batch_offline['done']).to(device)
                batch = concatenate_batch_transitions(
                    left_batch_transitions=batch, right_batch_transition=batch_offline
                )

        check_nan_in_transition(observations=batch["state"], actions=batch["action"], next_state=batch["next_state"])

        observation_features, next_observation_features = get_observation_features(
            policy=policy, observations=batch["state"], next_observations=batch["next_state"]
        )

        # Create a batch dictionary with all required elements for the forward method
        forward_batch = batch
        forward_batch['observation_feature'] = observation_features
        forward_batch['next_observation_feature'] = next_observation_features


        training_infos = {}
        if "critic" in optimizers.keys():
            critic_output = policy.forward(forward_batch, model="critic")

            loss_critic = critic_output.pop("loss_critic")

            optimizers["critic"].zero_grad()
            loss_critic.backward()
            training_infos["loss_critic"] = loss_critic.item()
            if hasattr(policy, "critic_ensemble"):
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
                )
            elif hasattr(policy, "critic_qc"):
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.critic_qc.parameters(), max_norm=clip_grad_norm_value
                )
            optimizers["critic"].step()

            # Initialize training info dictionary
            training_infos["critic_grad_norm"] = critic_grad_norm.mean().item()

            training_infos.update(critic_output)

        # Discrete critic optimization (if available)
        if "discrete_critic" in optimizers.keys():
            discrete_critic_output = policy.forward(forward_batch, model="discrete_critic")
            loss_discrete_critic = discrete_critic_output.pop("loss_discrete_critic")
            optimizers["discrete_critic"].zero_grad()
            loss_discrete_critic.backward()
            discrete_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value
            ).item()
            optimizers["discrete_critic"].step()

            # Add discrete critic info to training info
            training_infos["loss_discrete_critic"] = loss_discrete_critic.item()
            training_infos["discrete_critic_grad_norm"] = discrete_critic_grad_norm
            training_infos.update(discrete_critic_output)



        # Actor and temperature optimization (at specified frequency)
        if optimization_step % policy_update_freq == 0:
            for _ in range(policy_update_freq):
                # Actor optimization
                actor_output = policy.forward(forward_batch, model="actor")
                # if hasattr(policy.actor, "mean_layer"):
                #     last_layer_modules = [policy.actor.mean_layer]
                #     if hasattr(policy.actor, "std_layer"):
                #         last_layer_modules.append(policy.actor.std_layer)
                #     last_layer_params = [
                #         p for m in last_layer_modules for p in m.parameters() if p.requires_grad
                #     ]

                #     def _grad_norm(loss, retain):
                #         grads = torch.autograd.grad(loss, last_layer_params, retain_graph=retain, allow_unused=True)
                #         gs = [g for g in grads if g is not None]
                #         return torch.norm(torch.stack([g.norm() for g in gs])).item()

                #     # ── 测量各项损失的梯度贡献（不修改 .grad，不需要 zero_grad）──
                #     # donated_buffer=False 已在 modeling 中设置，retain_graph=True 可正常使用
                #     trace_keys = ['loss_from_q', 'loss_from_entropy']
                #     for key in trace_keys:
                #         if key in actor_output:
                #             trace_loss = actor_output.pop(key)
                #             new_key = key.split('_')[-1]
                #             grad_norm_from_key = _grad_norm(trace_loss, retain=True)
                #             training_infos[f"grad_norm_from_{new_key}"] = grad_norm_from_key

                loss_actor = actor_output.pop("loss_actor") 
                optimizers["actor"].zero_grad() # Reset the gradient buffers of the actor
                loss_actor.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                        parameters=policy.actor.parameters(), max_norm=clip_grad_norm_value
                    ).item()
                optimizers["actor"].step()

                # Add actor info to training info
                training_infos["loss_actor"] = loss_actor.item()
                training_infos["actor_grad_norm"] = actor_grad_norm
                training_infos.update(actor_output)
                

                if "lagrange" in optimizers.keys() and optimization_step % 1 == 0:
                    lagrange_output = policy.forward(forward_batch, model="lagrange")
                    loss_lagrange = lagrange_output.pop("loss_lagrange")

                    optimizers["lagrange"].zero_grad()
                    loss_lagrange.backward()
                    lagrange_grad_norm = torch.nn.utils.clip_grad_norm_(
                        parameters=policy.lagrange_net.parameters(), max_norm=clip_grad_norm_value
                    ).item()
                    optimizers["lagrange"].step()
                    training_infos["loss_lagrange"] = loss_lagrange.item()
                    training_infos["lagrange_grad_norm"] = lagrange_grad_norm
                    training_infos.update(lagrange_output)
            
                policy.update_target_networks()

                # # Temperature optimization
                if "temperature" in optimizers.keys():
                    temperature_output = policy.forward(forward_batch, model="temperature")
                    loss_temperature = temperature_output.pop("loss_temperature")
                    optimizers["temperature"].zero_grad()
                    loss_temperature.backward()
                    temp_grad_norm = torch.nn.utils.clip_grad_norm_(
                        parameters=[policy.log_alpha], max_norm=clip_grad_norm_value
                    ).item()
                    optimizers["temperature"].step()

                    # Add temperature info to training info
                    training_infos["loss_temperature"] = loss_temperature.item()
                    training_infos["temperature_grad_norm"] = temp_grad_norm
                    training_infos["temperature"] = policy.temperature
                    training_infos.update(temperature_output)

                    # Update temperature
                    policy.update_temperature()

        # Push policy to actors if needed
        # Push the latest policy parameters to the actor so it interacts with the updated policy
        if time.time() - last_time_policy_pushed > policy_parameters_push_frequency:
            push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)
            last_time_policy_pushed = time.time()

        # Update target networks (main and discrete)

        # Log training metrics at specified intervals
        if optimization_step % 5 == 0:
            training_infos["replay_buffer_size"] = len(replay_buffer)
            if offline_replay_buffer is not None:
                training_infos["offline_replay_buffer_size"] = len(offline_replay_buffer)
                
            training_infos["Optimization step"] = optimization_step

            # Log training metrics
            if wandb_logger:
                # print('======================== logging training_infos with wandb logger ====================')
                wandb_logger.log_dict(d=training_infos, mode="train", custom_step_key="Optimization step")

        # Calculate and log optimization frequency
        time_for_one_optimization_step = time.time() - time_for_one_optimization_step
        frequency_for_one_optimization_step = 1 / (time_for_one_optimization_step + 1e-9)

        # logging.info(f"[LEARNER] Optimization frequency loop [Hz]: {frequency_for_one_optimization_step}")

        # Log optimization frequency
        if wandb_logger:
            wandb_logger.log_dict(
                {
                    "Optimization frequency loop [Hz]": frequency_for_one_optimization_step,
                    "Optimization step": optimization_step,
                },
                mode="train",
                custom_step_key="Optimization step",
            )

        optimization_step += 1

        if saving_checkpoint and (optimization_step % save_freq == 0 or optimization_step == online_steps):
            print(f"Saving checkpoint at step {optimization_step}")
            save_training_checkpoint(
                cfg=cfg,
                optimization_step=optimization_step,
                online_steps=online_steps,
                interaction_message=interaction_message,
                policy=policy,
                optimizers=optimizers,
                replay_buffer=replay_buffer,
                offline_replay_buffer=offline_replay_buffer,
                dataset_repo_id=dataset_repo_id,
                fps=fps,
            )


def start_learner(
    parameters_queue: Queue,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    shutdown_event: any,  # Event,
    cfg: TrainRLServerPipelineConfig,
):
    """
    Start the learner server for training.
    It will receive transitions and interaction messages from the actor server,
    and send policy parameters to the actor server.

    Args:
        parameters_queue: Queue for sending policy parameters to the actor
        transition_queue: Queue for receiving transitions from the actor
        interaction_message_queue: Queue for receiving interaction messages from the actor
        shutdown_event: Event to signal shutdown
        cfg: Training configuration
    """
    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_process_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Learner server process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        # Return back for MP
        # TODO: Check if its useful
        _ = ProcessSignalHandler(False, display_pid=True)

    service = learner_service.LearnerService(
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        seconds_between_pushes=cfg.policy.actor_learner_config.policy_parameters_push_frequency,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        queue_get_timeout=cfg.policy.actor_learner_config.queue_get_timeout,
    )

    server = grpc.server(
        ThreadPoolExecutor(max_workers=learner_service.MAX_WORKERS),
        options=[
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
        ],
    )

    services_pb2_grpc.add_LearnerServiceServicer_to_server(
        service,
        server,
    )

    host = cfg.policy.actor_learner_config.learner_host
    port = cfg.policy.actor_learner_config.learner_port

    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logging.info("[LEARNER] gRPC server started")

    shutdown_event.wait()
    logging.info("[LEARNER] Stopping gRPC server...")
    server.stop(learner_service.SHUTDOWN_TIMEOUT)
    logging.info("[LEARNER] gRPC server stopped")



import tqdm





def offline_training(cfg: TrainRLServerPipelineConfig, policy: nn.Module, optimizers: dict, offline_replay_buffer: ReplayBuffer, wandb_logger: WandBLogger | None):
    pretrain_dir = os.path.join(os.getcwd(), "../../../", f"expert_model_{cfg.policy.type}")
    if not os.path.exists(pretrain_dir):
        os.makedirs(pretrain_dir)

    actor_pretrain_path = os.path.join(pretrain_dir, "actor_pretrain.pth")
    expert_pretrain_path = os.path.join(pretrain_dir, "expert_pretrain.pth")
    device = get_safe_torch_device(try_device=cfg.policy.device, log=True)
    if os.path.exists(expert_pretrain_path) and os.path.exists(actor_pretrain_path):
        # todo: debug
        policy.actor.load_state_dict(torch.load(actor_pretrain_path))
        policy.actor.train()
        policy.expert_network.load_state_dict(torch.load(expert_pretrain_path))
        policy.expert_network.train()

        if hasattr(policy, "actor_target"):
            policy.actor_target.load_state_dict(policy.actor.state_dict())
            policy.actor_target.eval()
        print(' success load actor pretrain model from ', actor_pretrain_path)
        return 


    batch_size = cfg.batch_size
    async_prefetch = cfg.policy.async_prefetch
    offline_iterator = None
    clip_grad_norm_value = cfg.policy.grad_clip_norm
    # NOTE: THIS IS THE MAIN LOOP OF THE LEARNER
    offline_iterator = offline_replay_buffer.get_iterator(
            batch_size=batch_size, async_prefetch=async_prefetch, queue_size=2
        )
    global expert_training_step
    for optimization_step in tqdm.tqdm(range(500), desc="Offline training"):
        batch_offline = next(offline_iterator)
        batch_offline['is_intervention'] = torch.ones_like(batch_offline['done']).to(device)
        batch = batch_offline
        actions = batch["action"]
        rewards = batch["reward"]
        observations = batch["state"]
        next_observations = batch["next_state"]
        done = batch["done"]
        is_intervention = batch["is_intervention"]


        check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)

        observation_features, next_observation_features = get_observation_features(
            policy=policy, observations=observations, next_observations=next_observations
        )

        # Create a batch dictionary with all required elements for the forward method
        forward_batch = {
            "action": actions,
            "reward": rewards,
            "state": observations,
            "next_state": next_observations,
            "done": done,
            "observation_feature": observation_features,
            "next_observation_feature": next_observation_features,
            "is_intervention": is_intervention,
            "complementary_info": batch["complementary_info"],
        }
    
        expert_output = policy.forward(forward_batch, model="expert")
        loss_expert = expert_output["loss_expert"] 
        optimizers["expert"].zero_grad()
        loss_expert.backward()
        expert_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=policy.expert_network.parameters(), max_norm=clip_grad_norm_value
        ).item()
        optimizers["expert"].step()
        actor_output = policy.forward(forward_batch, model="actor_bc")
        loss_actor_bc = actor_output["loss_actor_bc"] 
        # actor_output = policy.forward(forward_batch, model="actor")
        # loss_actor_bc = actor_output["loss_actor"]
        optimizers["actor"].zero_grad()
        loss_actor_bc.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=policy.actor.parameters(), max_norm=clip_grad_norm_value
        ).item()
        optimizers["actor"].step()


        training_infos = {}
        # Add actor info to training info
        training_infos["loss_expert"] = loss_expert.item()
        training_infos["allow_d"] = expert_output.get("allow_d", 0.0)
        training_infos["loss_actor_bc"] = loss_actor_bc.item()


        # Log training metrics at specified intervals
        # if optimization_step % 10 == 0:
        training_infos["expert_training_step"] = expert_training_step
        if wandb_logger:
            wandb_logger.log_dict(d=training_infos, mode="expert", custom_step_key="expert_training_step")
        
        expert_training_step += 1



    # print('expert_pretrain_path:', expert_pretrain_path)
    torch.save(policy.expert_network.state_dict(), expert_pretrain_path)
    torch.save(policy.actor.state_dict(), actor_pretrain_path) 
    print(' success save actor pretrain model to ', actor_pretrain_path)
    # torch.save(policy.critic_ensemble.state_dict(), critic_pretrain_path)
    if hasattr(policy, "actor_target"):
        policy.actor_target.load_state_dict(policy.actor.state_dict())
        policy.actor_target.eval()



def expert_training(offline_replay_buffer, optimizers, policy, clip_grad_norm_value, device, async_prefetch, wandb_logger, optimization_step):
    batch_size = 256
    
    global expert_training_step

   
    offline_iterator = offline_replay_buffer.get_iterator(batch_size=batch_size, async_prefetch=async_prefetch, queue_size=2)

    training_infos = {}
    print(' >>> begin expert update, expert_training_step:', expert_training_step)
    for _ in tqdm.tqdm(range(50), desc="Expert training"):
        batch_offline = next(offline_iterator)
        batch_offline['is_intervention'] = torch.ones_like(batch_offline['done']).to(device)
        batch = batch_offline
        actions = batch["action"]
        rewards = batch["reward"]
        observations = batch["state"]
        next_observations = batch["next_state"]
        done = batch["done"]
        is_intervention = batch["is_intervention"]
        
        check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)
        observation_features, next_observation_features = get_observation_features(
            policy=policy, observations=observations, next_observations=next_observations
        )
        forward_batch = {
            "action": actions,
            "reward": rewards,
            "state": observations,
            "next_state": next_observations,
            "done": done,
            "observation_feature": observation_features,
            "next_observation_feature": next_observation_features,
            "is_intervention": is_intervention,
            "complementary_info": batch["complementary_info"],
        }
        expert_output= policy.forward(forward_batch, model="expert")
        loss_expert = expert_output["loss_expert"] 
        allow_distance = expert_output.get("allow_d", 0.0)
        optimizers["expert"].zero_grad()
        loss_expert.backward()
        expert_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=policy.expert_network.parameters(), max_norm=clip_grad_norm_value
        ).item()
        optimizers["expert"].step()

        training_infos["loss_expert"] = loss_expert.item()
        training_infos["allow_d"] = allow_distance
        training_infos["expert_grad_norm"] = expert_grad_norm

        training_infos["expert_training_step"] = expert_training_step
        print(f'----------- in {expert_training_step} allow_d: {allow_distance}')

        if wandb_logger:
            wandb_logger.log_dict(d=training_infos, custom_step_key="expert_training_step", mode="expert")

        expert_training_step += 1

      

def save_training_checkpoint(
    cfg: TrainRLServerPipelineConfig,
    optimization_step: int,
    online_steps: int,
    interaction_message: dict | None,
    policy: nn.Module,
    optimizers: dict[str, Optimizer],
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer | None = None,
    dataset_repo_id: str | None = None,
    fps: int = 30,
) -> None:
    # Log the step this checkpoint is saved at, for debugging and monitoring
    logging.info(f"Checkpoint policy after step {optimization_step}")
    
    # Minimum digit count for step numbers, so directory names stay aligned (e.g. 000001, 000100)
    _num_digits = max(6, len(str(online_steps)))
    
    # Current interaction step (defaults to 0 if no interaction message was received); used to align progress when resuming
    interaction_step = interaction_message["Interaction step"] if interaction_message is not None else 0

    # 1. Create the checkpoint root directory under the Hydra run dir:
    #    experiments/<task>/exp_local/<date>/<overrides>/<seed>/checkpoints/step_xxx
    output_dir = Path(cfg.output_dir)
    checkpoint_dir = get_step_checkpoint_dir(output_dir, online_steps, optimization_step)
    
    # 2. Model save path: <checkpoint_dir>/pretrained_model/<current step>
    model_dir = os.path.join(checkpoint_dir, PRETRAINED_MODEL_DIR, str(optimization_step))
    
    # Save the model weights and config
    # Stores the state_dict and config files so the policy can be reloaded with from_pretrained
    policy.save_pretrained(model_dir)
    print('save model under', model_dir)

    if "hgdagger" not in cfg.policy.type:
        # (Disabled alternative) save a full checkpoint including optimizer and scheduler state
        # Uncomment this block to restore the previous optimizer state when resuming
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            step=optimization_step,
            cfg=cfg,
            policy=policy,
            optimizer=optimizers,
            scheduler=None,  # This pipeline uses no LR scheduler, so pass None
        )

    # 3. Save the training state (optimization step + interaction step)
    training_state_dir = os.path.join(checkpoint_dir, TRAINING_STATE_DIR)
    os.makedirs(training_state_dir, exist_ok=True)  # Create the directory if it does not exist
    
    # Training state: the progress information required to resume training
    training_state = {
        "step": optimization_step,  # Optimization step; training resumes from here
        "interaction_step": interaction_step  # Interaction step, aligned with the actor's progress
    }
    # Write the training state to disk
    torch.save(training_state, os.path.join(training_state_dir, "training_state.pt"))

    # 4. Update the "last" symlink to point at the newest checkpoint directory
    # This gives quick access to the latest model without knowing its step number
    update_last_checkpoint(checkpoint_dir)

    # 5. Save the online replay buffer as a standard dataset (temporary; may move to the robot side later)
    # Dataset path: <hydra_run_dir>/dataset
    dataset_dir = os.path.join(output_dir, "dataset")
    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)

    # Dataset repo id: prefer the given dataset_repo_id, otherwise fall back to the env task name
    repo_id_buffer_save = cfg.env.task if dataset_repo_id is None else dataset_repo_id
    
    # Convert the replay buffer into the standard LeRobot dataset format for later reuse
    replay_buffer.to_lerobot_dataset(
        repo_id=repo_id_buffer_save,  # Dataset identifier
        fps=fps,  # Matches the env frame rate to keep timing consistent
        root=dataset_dir  # Save root directory
    )

    # 6. Save the offline replay buffer as a separate dataset
    if offline_replay_buffer is not None:
        # Offline dataset path: <hydra_run_dir>/dataset_offline
        dataset_offline_dir = os.path.join(output_dir, "dataset_offline")
        
        # Remove stale data if the offline dataset directory already exists
        if os.path.exists(dataset_offline_dir) and os.path.isdir(dataset_offline_dir):
            shutil.rmtree(dataset_offline_dir)

        # Save the offline buffer as a standard dataset, keyed by the offline repo_id
        offline_replay_buffer.to_lerobot_dataset(
            repo_id_buffer_save,  # Offline dataset repo id, taken from the config
            fps=fps,  # Matches the env frame rate
            root=dataset_offline_dir  # Offline dataset save root directory
        )

    # Log completion and note that training can be resumed
    logging.info("Resume training")


def apply_sim_task_overrides(cfg, env_cfg):
    features = {}
    features_map = {}
    for cam in env_cfg.lerobot.cameras:
        key = f"observation.images.{cam}"
        features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(128, 128, 3))
        features_map[key] = key
    features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(env_cfg.lerobot.state_dim,))
    features_map["observation.state"] = "observation.state"
    features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=(env_cfg.lerobot.action_dim,))
    features_map["action"] = "action"

    cfg.job_name = env_cfg.task_name
    cfg.env.fps = env_cfg.lerobot.fps
    cfg.env.features = features
    cfg.env.features_map = features_map
    cfg.policy.num_discrete_actions = env_cfg.lerobot.num_discrete_actions

    policy_input_features = {}
    for cam in env_cfg.lerobot.cameras:
        policy_input_features[f"observation.images.{cam}"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 128, 128)
        )
    policy_input_features["observation.state"] = PolicyFeature(
        type=FeatureType.STATE, shape=(env_cfg.lerobot.state_dim,)
    )
    cfg.policy.input_features = policy_input_features



#################################################
# Training setup functions #
#################################################


def handle_resume_logic(cfg: TrainRLServerPipelineConfig) -> TrainRLServerPipelineConfig:
    """
    Handle the resume logic for training.

    If resume is True:
    - Verifies that a checkpoint exists
    - Loads the checkpoint configuration
    - Logs resumption details
    - Returns the checkpoint configuration

    If resume is False:
    - Checks if an output directory exists (to prevent accidental overwriting)
    - Returns the original configuration

    Args:
        cfg (TrainRLServerPipelineConfig): The training configuration

    Returns:
        TrainRLServerPipelineConfig: The updated configuration

    Raises:
        RuntimeError: If resume is True but no checkpoint found, or if resume is False but directory exists
    """
    out_dir = cfg.output_dir

    # Case 1: Not resuming, but need to check if directory exists to prevent overwrites
    if not cfg.resume:
        checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
        if os.path.exists(checkpoint_dir):
            raise RuntimeError(
                f"Output directory {checkpoint_dir} already exists. Use `resume=true` to resume training."
            )
        return cfg

    # Case 2: Resuming training
    checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)

    if not os.path.exists(checkpoint_dir):
        raise RuntimeError(f"No model checkpoint found in {checkpoint_dir} for resume=True")

    # Log that we found a valid checkpoint and are resuming
    logging.info(
        colored(
            "Valid checkpoint found: resume=True detected, resuming previous run",
            color="yellow",
            attrs=["bold"],
        )
    )

    # Load config using Draccus
    checkpoint_cfg_path = os.path.join(checkpoint_dir, PRETRAINED_MODEL_DIR)

    checkpoint_cfg = TrainRLServerPipelineConfig.from_pretrained(checkpoint_cfg_path)

    # Ensure resume flag is set in returned config
    checkpoint_cfg.resume = True
    # todo: debug
    # checkpoint_cfg.output_dir = out_dir
    return checkpoint_cfg


def load_training_state(
    cfg: TrainRLServerPipelineConfig,
    optimizers: Optimizer | dict[str, Optimizer],
):
    """
    Loads the training state (optimizers, step count, etc.) from a checkpoint.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        optimizers (Optimizer | dict): Optimizers to load state into

    Returns:
        tuple: (optimization_step, interaction_step) or (None, None) if not resuming
    """
    if not cfg.resume:
        return None, None

    # Construct path to the last checkpoint directory
    checkpoint_dir = os.path.join(cfg.output_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
    print('------------------------------checkpoint_dir', checkpoint_dir)

    logging.info(f"Loading training state from {checkpoint_dir}")

    try:
        # Use the utility function from train_utils which loads the optimizer state
        step, optimizers, _ = utils_load_training_state(Path(checkpoint_dir), optimizers, None)

        # Load interaction step separately from training_state.pt
        training_state_path = os.path.join(checkpoint_dir, TRAINING_STATE_DIR, "training_state.pt")
        interaction_step = 0
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, weights_only=False)  # nosec B614: Safe usage of torch.load
            interaction_step = training_state.get("interaction_step", 0)

        logging.info(f"Resuming from step {step}, interaction step {interaction_step}")
        return step, interaction_step

    except Exception as e:
        logging.error(f"Failed to load training state: {e}")
        traceback.print_exc()
        exit(-1)
        return None, None


def log_training_info(cfg: TrainRLServerPipelineConfig, policy: nn.Module) -> None:
    """
    Log information about the training process.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        policy (nn.Module): Policy model
    """
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.policy.online_steps=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")


def initialize_replay_buffer(
    cfg: TrainRLServerPipelineConfig, device: str, storage_device: str
) -> ReplayBuffer:
    """
    Initialize a replay buffer, either empty or from a dataset if resuming.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        device (str): Device to store tensors on
        storage_device (str): Device for storage optimization

    Returns:
        ReplayBuffer: Initialized replay buffer
    """
    if not cfg.resume:
        return ReplayBuffer(
            capacity=cfg.policy.online_buffer_capacity,
            device=device,
            state_keys=cfg.policy.input_features.keys(),
            storage_device=storage_device,
            optimize_memory=True,
        )

    logging.info("Resume training load the online dataset")
    dataset_path = os.path.join(cfg.output_dir, "dataset")

    # NOTE: In RL is possible to not have a dataset.
    repo_id = None
    if cfg.dataset is not None:
        repo_id = cfg.dataset.repo_id
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_path,
    )
    return ReplayBuffer.from_lerobot_dataset(
        lerobot_dataset=dataset,
        capacity=cfg.policy.online_buffer_capacity,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        optimize_memory=True
    )

import traceback
import sys

def initialize_offline_replay_buffer(
    cfg: TrainRLServerPipelineConfig,
    device: str,
    storage_device: str,
) -> ReplayBuffer:
    """
    Initialize an offline replay buffer from a dataset.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        device (str): Device to store tensors on
        storage_device (str): Device for storage optimization

    Returns:
        ReplayBuffer: Initialized offline replay buffer
    """
    if cfg.dataset is None:
        return ReplayBuffer(
            capacity=cfg.policy.offline_buffer_capacity,
            device=device,
            state_keys=cfg.policy.input_features.keys(),
            storage_device=storage_device,
            optimize_memory=True,
        )
        
    if not cfg.resume:
        logging.info("make_dataset offline buffer")
        # INSERT_YOUR_CODE
        offline_dataset = make_dataset(cfg)
    else:
        logging.info("load offline dataset")
        dataset_offline_path = os.path.join(cfg.output_dir, "dataset_offline")
        offline_dataset = LeRobotDataset(
            repo_id=cfg.dataset.repo_id,
            root=dataset_offline_path,
        )
        
    logging.info("Convert to a offline replay buffer")
    try:
        offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
            offline_dataset,
            device=device,
            state_keys=cfg.policy.input_features.keys(),
            storage_device=storage_device,
            optimize_memory=True,
            capacity=cfg.policy.offline_buffer_capacity
        )
    except Exception as e:
        print(f"[{type(e).__name__}] {e!r}")
        traceback.print_exc()          # full stacktrace
        sys.exit(1)
    return offline_replay_buffer


#################################################
# Utilities/Helpers functions #
#################################################


def get_observation_features(
    policy, observations: torch.Tensor, next_observations: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Get observation features from the policy encoder. It act as cache for the observation features.
    when the encoder is frozen, the observation features are not updated.
    We can save compute by caching the observation features.

    Args:
        policy: The policy model
        observations: The current observations
        next_observations: The next observations

    Returns:
        tuple: observation_features, next_observation_features
    """

    if policy.config.vision_encoder_name is None or not policy.config.freeze_vision_encoder:
        return None, None

    with torch.no_grad():
        observation_features = policy.actor.encoder.get_cached_image_features(observations, normalize=True)
        next_observation_features = policy.actor.encoder.get_cached_image_features(
            next_observations, normalize=True
        )


    return observation_features, next_observation_features


def use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    return cfg.policy.concurrency.learner == "threads"


def check_nan_in_transition(
    observations: torch.Tensor,
    actions: torch.Tensor,
    next_state: torch.Tensor,
    raise_error: bool = False,
) -> bool:
    """
    Check for NaN values in transition data.

    Args:
        observations: Dictionary of observation tensors
        actions: Action tensor
        next_state: Dictionary of next state tensors
        raise_error: If True, raises ValueError when NaN is detected

    Returns:
        bool: True if NaN values were detected, False otherwise
    """
    nan_detected = False

    # Check observations
    for key, tensor in observations.items():
        if torch.isnan(tensor).any():
            logging.error(f"observations[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in observations[{key}]")

    # Check next state
    for key, tensor in next_state.items():
        if torch.isnan(tensor).any():
            logging.error(f"next_state[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in next_state[{key}]")

    # Check actions
    if torch.isnan(actions).any():
        logging.error("actions contains NaN values")
        nan_detected = True
        if raise_error:
            raise ValueError("NaN detected in actions")

    return nan_detected


def push_actor_policy_to_queue(parameters_queue: Queue, policy: nn.Module):
    logging.debug("[LEARNER] Pushing actor policy to the queue")

    # Create a dictionary to hold all the state dicts
    state_dicts = {"policy": move_state_dict_to_device(policy.actor.state_dict(), device="cpu")}

    # Add discrete critic if it exists
    if hasattr(policy, "discrete_critic") and policy.discrete_critic is not None:
        state_dicts["discrete_critic"] = move_state_dict_to_device(
            policy.discrete_critic.state_dict(), device="cpu"
        )
        logging.debug("[LEARNER] Including discrete critic in state dict push")
    
    if hasattr(policy, "discrete_actor") and policy.discrete_actor is not None:
        state_dicts["discrete_actor"] = move_state_dict_to_device(
            policy.discrete_actor.state_dict(), device="cpu"
        )
        logging.debug("[LEARNER] Including discrete actor in state dict push")
    
    if hasattr(policy, "critic_qc") and policy.critic_qc is not None:
        state_dicts["critic_qc"] = move_state_dict_to_device(
            policy.critic_qc.state_dict(), device="cpu"
        )
        logging.debug("[LEARNER] Including critic_qc in state dict push")

    state_bytes = state_to_bytes(state_dicts)
    parameters_queue.put(state_bytes)


def process_interaction_message(
    message, interaction_step_shift: int, wandb_logger: WandBLogger | None = None
):
    """Process a single interaction message with consistent handling."""
    message = bytes_to_python_object(message)
    # Shift interaction step for consistency with checkpointed state
    message["Interaction step"] += interaction_step_shift

    # Log if logger available
    if wandb_logger:
        wandb_logger.log_dict(d=message, mode="train", custom_step_key="Interaction step")

    return message


def process_transitions(
    optimization_step: int,
    transition_queue: Queue,
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer,
    device: str,
    dataset_repo_id: str | None,
    shutdown_event: any,
    optimizers: dict[str, Optimizer],
    policy: nn.Module,
    clip_grad_norm_value: float,
    batch_size: int,
    async_prefetch: bool,
    wandb_logger: WandBLogger | None,
    cfg: TrainRLServerPipelineConfig,
):
    """Process all available transitions from the queue.

    Args:
        transition_queue: Queue for receiving transitions from the actor
        replay_buffer: Replay buffer to add transitions to
        offline_replay_buffer: Offline replay buffer to add transitions to
        device: Device to move transitions to
        dataset_repo_id: Repository ID for dataset
        shutdown_event: Event to signal shutdown
    """
    global new_offline_transition_num
    while not transition_queue.empty() and not shutdown_event.is_set():
        transition_list = transition_queue.get()
        transition_list = bytes_to_transitions(buffer=transition_list)
        new_transition_num = 0
        
        success_list = []
        reward_list = []


        for transition in transition_list:
            transition['complementary_info']['target_prob'] = 0
            transition = move_transition_to_device(transition=transition, device=device)

            # Skip transitions with NaN values
            if check_nan_in_transition(
                observations=transition["state"],
                actions=transition["action"],
                next_state=transition["next_state"],
            ):
                logging.warning("[LEARNER] NaN detected in transition, skipping")
                continue

            # Add all valid data to the main online buffer
            replay_buffer.add(**transition)
            # Add data with intervention to the offline buffer
            if offline_replay_buffer is not None and (transition.get("complementary_info", {}).get(
                "is_intervention") or transition.get("complementary_info", {}).get(
                "first_intervene_reward") < 0 ):
                offline_replay_buffer.add(**transition)
                new_offline_transition_num += 1
                if new_offline_transition_num % 50 == 0 and "silri" in cfg.policy.type:
                    expert_training(offline_replay_buffer, optimizers, policy, clip_grad_norm_value, device, async_prefetch, wandb_logger, optimization_step)
                
            new_transition_num += 1


def process_interaction_messages(
    interaction_message_queue: Queue,
    interaction_step_shift: int,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,
) -> dict | None:
    """Process all available interaction messages from the queue.

    Args:
        interaction_message_queue: Queue for receiving interaction messages
        interaction_step_shift: Amount to shift interaction step by
        wandb_logger: Logger for tracking progress
        shutdown_event: Event to signal shutdown

    Returns:
        dict | None: The last interaction message processed, or None if none were processed
    """
    last_message = None
    while not interaction_message_queue.empty() and not shutdown_event.is_set():
        message = interaction_message_queue.get()
        last_message = process_interaction_message(
            message=message,
            interaction_step_shift=interaction_step_shift,
            wandb_logger=wandb_logger,
        )

    return last_message


if __name__ == "__main__":
    try:
        if '--g2-software' in sys.argv:
            from g2_local.runtime import main
            main('learner', [arg for arg in sys.argv[1:] if arg != '--g2-software'])
        else:
            train_cli()
        logging.info("[LEARNER] main finished")
    except Exception as e:
        print(f"[{type(e).__name__}] {e!r}")
        traceback.print_exc()          # full stacktrace
        sys.exit(1)
