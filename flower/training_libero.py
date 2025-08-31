import logging
from pathlib import Path
import sys
sys.tracebacklimit = None
import os
import wandb
import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
import copy
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, Optional, Tuple
import pytorch_lightning as pl

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

import flower.models.flower as models_m
from flower.models.q_networks import DoubleQNetwork
from flower.models.replay_buffer import SequenceReplayBuffer
from flower.utils.utils import get_git_commit_hash, get_last_checkpoint, initialize_pretrained_weights, print_system_env_info

# Add local repo to path
sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)



logger = logging.getLogger(__name__)

class QChunkingTrainer:
    """
    Q-Chunking trainer that implements the Q-chunking reinforcement learning algorithm.

    This trainer manages:
    - BC VLA policy (for behavior cloning)
    - RL VLA policy (identical to BC, for Q-learning)
    - Q-networks (double Q-networks for value estimation)
    - Target Q-networks (for stable training)
    - Replay buffer (for experience replay)

    The training process includes:
    1. BC flow matching loss (behavior cloning)
    2. Q-value loss (Q-learning)
    3. Distillation loss (between BC and RL policies)
    """

    def __init__(self, cfg: DictConfig, device: torch.device = None, env=None):
        self.cfg = cfg
        # Use devices from config if available

        self.device = device
        self.q_cfg = cfg.q_chunking
        self.env = env  # Environment for online training

        # Initialize models
        self.bc_vla = None
        self.rl_vla = None
        self.q_networks = None
        self.target_q_networks = None

        # Initialize replay buffer
        self.replay_buffer = None

        # Optimizers
        self.bc_optimizer = None
        self.rl_optimizer = None
        self.q_optimizer = None

        # Training state
        self.step_count = 0
        self.epoch = 0
        self.global_step = 0
        self.total_bc_loss = 0.0
        self.total_q_loss = 0.0
        self.total_distill_loss = 0.0

        # Training mode management
        self.training_stage = self.q_cfg.training_stage
        self.offline_steps = 0
        self.online_steps = 0

        logger.info(f"Initialized Q-Chunking trainer with stage: {self.training_stage}")

    def setup_models(self, base_model_cfg: DictConfig):
        """Initialize all models for Q-chunking."""
        logger.info("Setting up Q-chunking models...")

        # 1. BC VLA (behavior cloning policy) with 4 sampling steps for high-quality actions
        self.bc_vla = hydra.utils.instantiate(self.q_cfg.models.bc_vla, **base_model_cfg).to(self.device)

        # Load pretrained weights if available
        if "pretrain_chk" in self.cfg and self.cfg.pretrain_chk:
            from flower.utils.utils import initialize_pretrained_weights
            initialize_pretrained_weights(self.bc_vla, self.cfg)
            logger.info(f"Loaded pretrained weights for BC VLA from {self.cfg.pretrain_chk}")

        logger.info(f"Initialized BC VLA with {self.bc_vla.num_sampling_steps} sampling steps")

        # 2. RL VLA (identical to BC VLA, used for Q-learning) with 1 sampling step for fast online training
        self.rl_vla = hydra.utils.instantiate(self.q_cfg.models.rl_vla, **base_model_cfg).to(self.device)

        logger.info(f"Initialized RL VLA with {self.rl_vla.num_sampling_steps} sampling step")

        # 3. Q-networks (DoubleQNetwork already contains two Q-networks internally)
        self.q_networks = DoubleQNetwork(
            state_dim=self.q_cfg.q_network.state_dim,
            action_dim=self.q_cfg.q_network.action_dim,
            chunk_size=self.q_cfg.chunk_size,
            hidden_dim=self.q_cfg.q_network.hidden_dim,
            n_layers=self.q_cfg.q_network.n_layers,
            dropout=self.q_cfg.q_network.dropout
        ).to(self.device)
        logger.info("Initialized Double Q-networks")

        # 4. Target Q-networks (copy for stable training)
        self.target_q_networks = copy.deepcopy(self.q_networks)
        logger.info("Initialized target Q-networks")

        # 5. Setup replay buffer
        self.replay_buffer = SequenceReplayBuffer(
            capacity=self.q_cfg.replay_buffer_size,
            chunk_size=self.q_cfg.chunk_size,
            device=self.device
        )
        logger.info("Initialized replay buffer")


    def setup_optimizers(self):
        """Setup optimizers for all models using FlowerVLA's built-in method."""
        # Use FlowerVLA's built-in optimizer configuration
        # FlowerVLA.configure_optimizers() returns a dict with "optimizer" key
        bc_optim_config = self.bc_vla.configure_optimizers()
        self.bc_optimizer = bc_optim_config["optimizer"]

        rl_optim_config = self.rl_vla.configure_optimizers()
        self.rl_optimizer = rl_optim_config["optimizer"]

        # Q-networks optimizer
        self.q_optimizer = optim.AdamW(
            self.q_networks.parameters(),
            lr=self.q_cfg.q_lr,
            betas=self.cfg.model.optimizer.betas,
            weight_decay=0.01
        )

        logger.info("Setup optimizers")
    # 应该是对的
    def compute_bc_flow_matching_loss(self, batch: Dict[str, Any], model) -> torch.Tensor:
        """Compute behavior cloning flow matching loss using original VLA interface."""
        total_loss = 0.0

        # Handle original dataloader format: {"lang": {dataset_batch}}

        if isinstance(batch, dict) and all(isinstance(v, dict)  for v in batch.values()):
            #这个v取出的是“lang”的一个batch
            # Original batch format from dataloader
            for modality_scope, dataset_batch in batch.items():
                #
                if isinstance(dataset_batch, dict) and 'actions' in dataset_batch:
                    # Use original VLA interface exactly as designed
                    model.modality_scope = modality_scope
                    obs_features = model.encode_observations(dataset_batch)
                    loss, _ = model.rf_loss(obs_features, dataset_batch["actions"])
                    total_loss += loss
            total_loss=total_loss/len(batch)
        else:
            # Direct dataset_batch format (for replay buffer data)
            # Need to wrap in expected format
            if 'actions'  not in batch:
                raise ValueError(f"Invalid batch format - no actions found in {list(batch.keys())}")
            # Use original VLA interface
            dataset_batch = batch
            model.modality_scope = "lang"  # Default modality
            obs_features = model.encode_observations(dataset_batch)
            loss, _ = model.rf_loss(obs_features, dataset_batch["actions"])
            total_loss = loss


        return total_loss


    def compute_q_value_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute Q-value loss following original ACFQL algorithm exactly.
        Using the LIBERO data format with next_observations from your modifications.
        """

        if isinstance(batch,dict) and all(isinstance(v,dict) and 'actions' in v for v in batch.values()):
            modality_scope,dataset_batch=next(iter(batch.items()))
            self.rl_vla.modality_scope=modality_scope
        else:
            dataset_batch=batch
            self.rl_vla.modality_scope='lang'

        # Extract data from LIBERO batch format (following your data structure)
        actions = dataset_batch['actions'].to(self.device)
        rewards = dataset_batch['rewards'].to(self.device)
        dones = dataset_batch['dones'].to(self.device)


        # Flatten action chunks for Q-networks (following original algorithm)
        if self.q_cfg.action_chunking:
            # Reshape to (batch_size, chunk_size * action_dim)
            batch_actions = actions.reshape(actions.shape[0], -1)
        else:
            # Take first action only
            batch_actions = actions[..., 0, :]

        # 1. Sample next actions using RL policy (original line 39)
        # Create next_obs batch in same format as original dataloader
        if isinstance(batch, dict) and all(isinstance(v, dict) and 'actions' in v for v in batch.values()):
            next_obs_batch = {}
            for modality_scope,dataset_batch in batch.items():
                next_dataset_batch=dataset_batch.copy()
                next_dataset_batch['rgb_obs']=dataset_batch['next_rgb_obs']
                next_dataset_batch['robot_obs'] = dataset_batch['next_robot_obs']
                next_obs_batch[modality_scope] = next_dataset_batch
        else:
            next_obs_batch=dataset_batch.copy()
            next_obs_batch['rgb_obs'] = dataset_batch['next_rgb_obs']
            next_obs_batch['robot_obs'] = dataset_batch['next_robot_obs']



        with torch.no_grad():
            self.rl_vla.modality_scope = modality_scope if 'modality_scope' in locals() else 'lang'
            next_obs_features = self.rl_vla.encode_observations(next_obs_batch)

            # Sample next actions using original VLA interface
            noise = torch.randn_like(actions, device=self.device)
            next_actions_sampled = self.rl_vla.sample_actions(noise, next_obs_features, inference=True)

            # Flatten for Q-networks
            if self.q_cfg.action_chunking:
                next_actions_flat = next_actions_sampled.reshape(next_actions_sampled.shape[0], -1)
            else:
                next_actions_flat = next_actions_sampled[..., 0, :]

        # 2. Encode states for Q-networks (extract features from VLA)
        with torch.no_grad():
            current_obs_features = self.rl_vla.encode_observations(batch if isinstance(batch,dict) and all(isinstance(v,dict) and 'actions' in v for v in batch.values()) else dataset_batch)
            # Pool features to get state representation
            current_states = current_obs_features['features'].mean(dim=1)

            next_obs_features = self.rl_vla.encode_observations(next_obs_batch)
            next_states = next_obs_features['features'].mean(dim=1)

        # 3. Compute current Q-values (original line 50)
        q1_current, q2_current = self.q_networks(current_states, batch_actions)

        # 4. Compute target Q-values using target networks (original lines 41-45)
        with torch.no_grad():
            target_q1, target_q2 = self.target_q_networks(next_states, next_actions_flat)
            if self.q_cfg.q_aggregation == 'min':
                target_q = torch.min(target_q1, target_q2)
            else:
                target_q = (target_q1 + target_q2) / 2

        T=rewards.shape[1]

        H=min(T,self.cfg.act_seq_len)

        discount_pow=torch.arange(H, device=rewards.device, dtype=rewards.dtype)
        discount_chunk=torch.pow(self.q_cfg.discount_factor, discount_pow)


        chunk_rewards=torch.sum(rewards[:,:H]*discount_chunk.unsqueeze(0),dim=1,keepdim=True)

        if dones.dim()== 2:
             chunk_dones = dones[:, min(H-1, dones.shape[1]-1)].unsqueeze(1)
        else:
            chunk_dones=dones.unsqueeze(1)

        discount_H = self.q_cfg.discount_factor ** self.q_cfg.chunk_size
        td_target = chunk_rewards + discount_H * (1-chunk_dones) * target_q
        td_target=td_target.detach()

        # 6. MSE loss (original line 52)
        q1_loss = F.mse_loss(q1_current, td_target)
        q2_loss = F.mse_loss(q2_current, td_target)

        return q1_loss + q2_loss

    def compute_rl_actor_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute RL actor loss for RL VLA: Q maximization + Distillation (NO BC flow loss).
        RL VLA only learns from Q-values and distillation from BC VLA.
        """

        if isinstance(batch,dict) and all(isinstance(v,dict) and 'actions' in v for v in batch.values()):
            modality_scope,dataset_batch=next(iter(batch.items()))
            self.bc_vla.modality_scope = modality_scope
            self.rl_vla.modality_scope = modality_scope
        else:
            dataset_batch = batch
            self.bc_vla.modality_scope = 'lang'
            self.rl_vla.modality_scope = 'lang'

        actions = dataset_batch['actions'].to(self.device)

        # 1. Distillation loss - learn from BC VLA (target)
        with torch.no_grad():
            # Sample from BC policy (target flow actions)
            self.bc_vla.modality_scope = "lang"
            obs_features_bc = self.bc_vla.encode_observations(batch)
            noise = torch.randn_like(actions, device=self.device)
            target_flow_actions = self.bc_vla.sample_actions(noise, obs_features_bc, inference=True)

        # Sample from RL policy (actor actions)
        self.rl_vla.modality_scope = "lang"
        obs_features_rl = self.rl_vla.encode_observations(batch)
        actor_actions = self.rl_vla.sample_actions(noise, obs_features_rl, inference=False)

        # Distillation loss: make RL policy match BC policy
        distill_loss = F.mse_loss(actor_actions, target_flow_actions.detach())

        # 2. Q maximization loss - maximize Q-values for RL policy actions
        with torch.no_grad():
            current_states = obs_features_rl['features'].mean(dim=1)

        # Flatten actions for Q-network
        if self.q_cfg.action_chunking:
            actor_actions_flat = actor_actions.reshape(actor_actions.shape[0], -1)
        else:
            actor_actions_flat = actor_actions[..., 0, :]

        # Get Q-values and maximize them (negative Q loss)
        q1, q2 = self.q_networks(current_states, actor_actions_flat)
        q_mean = (q1 + q2) / 2  # Use mean of both Q-networks
        q_value = q_mean.mean()  # Maximize Q-values (negative loss)


        # Total RL actor loss: Distillation + Q maximization (NO BC flow loss)
        total_loss = self.q_cfg.distill_loss_weight * distill_loss + self.q_cfg.q_value_loss_weight *(q_value)*(-1)

        return total_loss,q_value

    def compute_distillation_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute distillation loss between BC and RL policies using original VLA interface."""
        total_distill_loss = 0.0


        # Handle batch format properly
        if isinstance(batch, dict) and all(isinstance(v, dict) and 'actions' in v for v in batch.values()):
            # Original dataloader format: {"lang": {dataset_batch}}
            for modality_scope, dataset_batch in batch.items():
                if isinstance(dataset_batch, dict) and 'actions' in dataset_batch:

                    # Use original VLA interface for both models
                    self.bc_vla.modality_scope = modality_scope
                    self.rl_vla.modality_scope = modality_scope

                    # Encode observations using original interface
                    obs_features_bc = self.bc_vla.encode_observations(dataset_batch)
                    obs_features_rl = self.rl_vla.encode_observations(dataset_batch)

                    # Sample noise with correct action dimensions
                    target_actions = dataset_batch["actions"]
                    noise = torch.randn_like(target_actions, device=self.device)

                    # Sample actions using original interface
                    bc_actions = self.bc_vla.sample_actions(noise.clone(), obs_features_bc, inference=True)
                    rl_actions = self.rl_vla.sample_actions(noise.clone(), obs_features_rl, inference=True)
                    bc_actions=bc_actions.detach()

                    # Compute distillation loss
                    distill_loss = F.mse_loss(rl_actions, bc_actions)
                    total_distill_loss += distill_loss

        else:
            # Direct dataset_batch format (for replay buffer)
            if 'actions' in batch:
                dataset_batch = batch

                # Use original VLA interface
                self.bc_vla.modality_scope = "lang"
                self.rl_vla.modality_scope = "lang"

                obs_features_bc = self.bc_vla.encode_observations(dataset_batch)
                obs_features_rl = self.rl_vla.encode_observations(dataset_batch)

                # Sample noise with correct dimensions
                target_actions = dataset_batch["actions"]
                noise = torch.randn_like(target_actions, device=self.device)

                bc_actions = self.bc_vla.sample_actions(noise.clone(), obs_features_bc, inference=True)
                rl_actions = self.rl_vla.sample_actions(noise.clone(), obs_features_rl, inference=True)
                bc_actions=bc_actions.detach()

                distill_loss = F.mse_loss(rl_actions, bc_actions)
                total_distill_loss = distill_loss

            else:
                raise ValueError(f"Invalid batch format - no actions found in {list(batch.keys())}")

        return total_distill_loss


    def _soft_update_target_networks(self):
        """Soft update target networks."""
        tau = self.q_cfg.tau

        for target_param, main_param in zip(
            self.target_q_networks.parameters(),
            self.q_networks.parameters()
        ):
            target_param.data.copy_(
                tau * main_param.data + (1.0 - tau) * target_param.data
            )


    def save_checkpoint(self, checkpoint_path: Path, checkpoint_type: str = "full"):
        """
        Save training checkpoint with specified type.
        Ensures compatibility with original evaluation script (flower_eval_libero.py).

        Args:
            checkpoint_path: Path to save checkpoint
            checkpoint_type: Type of checkpoint - 'bc_only', 'rl_only', 'full'
        """

        checkpoint_data = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'training_stage': self.training_stage,
            'offline_steps': self.offline_steps,
            'online_steps': self.online_steps,
            'pytorch-lightning_version': pl.__version__,
            'callbacks': {},
            'config': self.cfg
        }

        if checkpoint_type in ["bc_only", "full"]:
            # Save BC VLA with evaluation-compatible format
            checkpoint_data.update({
                'state_dict': self.bc_vla.state_dict(),  # Main model state_dict for evaluation
                'hyper_parameters': self.bc_vla.hparams,
                'bc_vla_state_dict': self.bc_vla.state_dict(),
                'bc_optimizer_state_dict': self.bc_optimizer.state_dict(),
            })

        if checkpoint_type in ["rl_only", "full"]:
            # For RL-only, set RL VLA as main model for evaluation
            if checkpoint_type == "rl_only":
                checkpoint_data.update({
                    'state_dict': self.rl_vla.state_dict(),  # Main model state_dict for evaluation
                    'hyper_parameters': self.rl_vla.hparams,
                })


            checkpoint_data.update({
                'rl_vla_state_dict': self.rl_vla.state_dict(),
                'rl_optimizer_state_dict': self.rl_optimizer.state_dict(),
            })

        if checkpoint_type == "full":
            # For full checkpoint, default to BC VLA for evaluation (more stable)
            if 'state_dict' not in checkpoint_data:
                checkpoint_data.update({
                    'state_dict': self.bc_vla.state_dict(),
                    'hyper_parameters': self.bc_vla.hparams,
                })
            checkpoint_data.update({
                'q_networks_state_dict': self.q_networks.state_dict(),
                'target_q_networks_state_dict': self.target_q_networks.state_dict(),
                'q_optimizer_state_dict': self.q_optimizer.state_dict(),
                'replay_buffer_state': self.replay_buffer.get_state() if self.replay_buffer else None,
            })

        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Saved {checkpoint_type} checkpoint: {checkpoint_path}")

        # Also save individual models in HuggingFace format using existing functions
        self._save_models_as_hf(checkpoint_path, checkpoint_type)

    def _save_models_as_hf(self, checkpoint_path: Path, checkpoint_type: str):
        """Save models in HuggingFace format compatible with evaluation script."""
        try:
            # Import dependencies
            try:
                from safetensors.torch import save_model
            except ImportError:
                logger.warning("safetensors not available, skipping HuggingFace format")
                return
            import json

            base_dir = checkpoint_path.parent / f"hf_models_{checkpoint_path.stem}"

            if checkpoint_type in ["bc_only", "full"]:
                # Save BC VLA in HuggingFace format for evaluation
                bc_dir = base_dir / "bc_vla"
                bc_dir.mkdir(parents=True, exist_ok=True)


                # Save model in both formats for compatibility
               # torch.save({'state_dict': self.bc_vla.state_dict()}, bc_dir / "model.pt")
                save_model(self.bc_vla, bc_dir / "model.safetensors")

                # Create config.yaml (not JSON) for evaluation script compatibility
                model_config = {
                    "model": {
                        "_target_": "flower.models.flower.FLOWERVLA",
                        "_recursive_": False,
                        "num_sampling_steps": getattr(self.bc_vla, 'num_sampling_steps', 4),
                        "lowdim_obs_dim": getattr(self.bc_vla, 'lowdim_obs_dim', None),
                        "action_dim": getattr(self.bc_vla, 'action_dim', None),
                        "act_window_size": getattr(self.bc_vla, 'act_window_size', None),
                        # Copy other essential parameters from hparams if available
                        **({k: v for k, v in self.bc_vla.hparams.items()
                           if k in ['vlm_path', 'dit_dim', 'n_heads', 'n_layers', 'use_second_view',
                                   'second_view_key', 'multistep', 'use_causal_attention', 'use_cross_attn']}
                          if hasattr(self.bc_vla, 'hparams') else {}),
                    },
                    "datamodule": {
                        "datasets": {
                            "lang_dataset": {
                                "lang_folder": "lang_clip_resnet50"
                            }
                        }
                    }
                }

                # Save as YAML for evaluation script compatibility
                with open(bc_dir / "config.yaml", 'w') as f:
                    OmegaConf.save(config=model_config, f=f)

                # Also create model card
                with open(bc_dir / "README.md", 'w') as f:
                    f.write(f"# FLOWER VLA BC Model\n\n")
                    f.write(f"Saved from Q-Chunking training checkpoint: {checkpoint_path.name}\n")
                    f.write(f"Model type: BC VLA with {getattr(self.bc_vla, 'num_sampling_steps', 4)} sampling steps\n")

            if checkpoint_type in ["rl_only", "full"]:
                # Save RL VLA in HuggingFace format for evaluation
                rl_dir = base_dir / "rl_vla"
                rl_dir.mkdir(parents=True, exist_ok=True)

                torch.save({'state_dict': self.rl_vla.state_dict()}, rl_dir / "model.pt")
                save_model(self.rl_vla, rl_dir / "model.safetensors")

                # Create config.yaml for evaluation script compatibility
                model_config = {
                    "model": {
                        "_target_": "flower.models.flower.FLOWERVLA",
                        "_recursive_": False,
                        "num_sampling_steps": getattr(self.rl_vla, 'num_sampling_steps', 1),
                        "lowdim_obs_dim": getattr(self.rl_vla, 'lowdim_obs_dim', None),
                        "action_dim": getattr(self.rl_vla, 'action_dim', None),
                        "act_window_size": getattr(self.rl_vla, 'act_window_size', None),
                        # Copy other essential parameters from hparams if available
                        **({k: v for k, v in self.rl_vla.hparams.items()
                           if k in ['vlm_path', 'dit_dim', 'n_heads', 'n_layers', 'use_second_view',
                                   'second_view_key', 'multistep', 'use_causal_attention', 'use_cross_attn']}
                          if hasattr(self.rl_vla, 'hparams') else {}),
                    },
                    "datamodule": {
                        "datasets": {
                            "lang_dataset": {
                                "lang_folder": "lang_clip_resnet50"
                            }
                        }
                    }
                }

                # Save as YAML for evaluation script compatibility
                with open(rl_dir / "config.yaml", 'w') as f:
                    OmegaConf.save(config=model_config, f=f)

                # Also create model card
                with open(rl_dir / "README.md", 'w') as f:
                    f.write(f"# FLOWER VLA RL Model\n\n")
                    f.write(f"Saved from Q-Chunking training checkpoint: {checkpoint_path.name}\n")
                    f.write(f"Model type: RL VLA with {getattr(self.rl_vla, 'num_sampling_steps', 1)} sampling step\n")
            '''
            if checkpoint_type in [ "full"]:
                # Save BC VLA in HuggingFace format for evaluation
                q_dir = base_dir / "q_networks"
                q_dir.mkdir(parents=True, exist_ok=True)

                # Save model in both formats for compatibility
               # torch.save({'state_dict': self.bc_vla.state_dict()}, bc_dir / "model.pt")
                save_model(self.q_networks, q_dir / "model.safetensors")

                # Create config.yaml (not JSON) for evaluation script compatibility
                model_config = {
                    "model": {
                        "_target_": "flower.models.q_networks.DoubleQNetwork",
                        "_recursive_": False,
                        "state_dim": 1024 , # VLA encoder output dimension

                        "hidden_dim": 512,
                        "n_layers": 3,
                        "dropout": 0.1,



                        # Copy other essential parameters from hparams if available
                        **({k: v for k, v in self.bc_vla.hparams.items()
                           if k in ['vlm_path', 'dit_dim', 'n_heads', 'n_layers', 'use_second_view',
                                   'second_view_key', 'multistep', 'use_causal_attention', 'use_cross_attn']}
                          if hasattr(self.bc_vla, 'hparams') else {}),
                    },
                    "datamodule": {
                        "datasets": {
                            "lang_dataset": {
                                "lang_folder": "lang_clip_resnet50"
                            }
                        }
                    }
                }

                # Save as YAML for evaluation script compatibility
                with open(bc_dir / "config.yaml", 'w') as f:
                    OmegaConf.save(config=model_config, f=f)

                # Also create model card
                with open(bc_dir / "README.md", 'w') as f:
                    f.write(f"# FLOWER VLA BC Model\n\n")
                    f.write(f"Saved from Q-Chunking training checkpoint: {checkpoint_path.name}\n")
                    f.write(f"Model type: BC VLA with {getattr(self.bc_vla, 'num_sampling_steps', 4)} sampling steps\n")
                '''

            logger.info(f"Saved models in HuggingFace format: {base_dir}")

        except Exception as e:
            logger.warning(f"Failed to save HuggingFace format: {e}")
            logger.warning(f"Error details: {str(e)}")


    def load_checkpoint(self, checkpoint_path: Path, load_type: str = "full"):
        """
        Load training checkpoint with specified type.
        Supports both PyTorch Lightning and HuggingFace format checkpoints.

        Args:
            checkpoint_path: Path to load checkpoint from
            load_type: Type of checkpoint to load - 'bc_only', 'rl_only', 'full'
        """
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return False

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

        # Load training state
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.step_count = checkpoint.get('step_count', 0)
        self.training_stage = checkpoint.get('training_stage', self.training_stage)
        self.offline_steps = checkpoint.get('offline_steps', 0)
        self.online_steps = checkpoint.get('online_steps', 0)

        # Load models based on load_type
        if load_type in ["bc_only", "full"]:
            if 'bc_vla_state_dict' in checkpoint:
                # Load from dedicated BC VLA state dict
                self.bc_vla.load_state_dict(checkpoint['bc_vla_state_dict'], strict=False)
                if 'bc_optimizer_state_dict' in checkpoint and self.bc_optimizer is not None:
                    try:
                        self.bc_optimizer.load_state_dict(checkpoint['bc_optimizer_state_dict'])
                    except Exception as e:
                        logger.warning(f"Failed to load BC optimizer state: {e}")
                logger.info("Loaded BC VLA checkpoint")
            elif load_type == "bc_only" and 'state_dict' in checkpoint:
                # Fallback: load from main state_dict for BC-only checkpoints
                self.bc_vla.load_state_dict(checkpoint['state_dict'], strict=False)
                logger.info("Loaded BC VLA from main state_dict")

        if load_type in ["rl_only", "full"]:
            if 'rl_vla_state_dict' in checkpoint:
                # Load from dedicated RL VLA state dict
                self.rl_vla.load_state_dict(checkpoint['rl_vla_state_dict'], strict=False)
                if 'rl_optimizer_state_dict' in checkpoint and self.rl_optimizer is not None:
                    try:
                        self.rl_optimizer.load_state_dict(checkpoint['rl_optimizer_state_dict'])
                    except Exception as e:
                        logger.warning(f"Failed to load RL optimizer state: {e}")
                logger.info("Loaded RL VLA checkpoint")
            elif load_type == "rl_only" and 'state_dict' in checkpoint:
                # Fallback: load from main state_dict for RL-only checkpoints
                self.rl_vla.load_state_dict(checkpoint['state_dict'], strict=False)
                logger.info("Loaded RL VLA from main state_dict")

        if load_type == "full":
            # Load Q-networks for full checkpoints
            if 'q_networks_state_dict' in checkpoint and self.q_networks is not None:
                try:
                    self.q_networks.load_state_dict(checkpoint['q_networks_state_dict'], strict=False)
                    if 'target_q_networks_state_dict' in checkpoint and self.target_q_networks is not None:
                        self.target_q_networks.load_state_dict(checkpoint['target_q_networks_state_dict'], strict=False)
                    if 'q_optimizer_state_dict' in checkpoint and self.q_optimizer is not None:
                        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
                    logger.info("Loaded Q-networks checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to load Q-networks: {e}")

            # Load replay buffer state
            if 'replay_buffer_state' in checkpoint and checkpoint['replay_buffer_state'] and self.replay_buffer is not None:
                try:
                    self.replay_buffer.load_state(checkpoint['replay_buffer_state'])
                    logger.info("Loaded replay buffer state")
                except Exception as e:
                    logger.warning(f"Failed to load replay buffer: {e}")

        logger.info(f"Successfully loaded {load_type} checkpoint from: {checkpoint_path}")
        return True


    def collect_online_transitions(self) -> int:
        """
        Collect transitions from environment using RL VLA policy.
        Following original Q-chunking algorithm exactly (main_online.py lines 184-252).

        Returns number of transitions collected in this episode.
        """
        if self.env is None:
            logger.warning("No environment provided for online trajectory collection")
            return 0

        obs = self.env.reset()
        done = False
        action_queue = []  # For action chunking (like original)
        transitions_collected = 0
        default_instruction = "manipulate object"

        while not done:
            # Sample action chunk if queue is empty (following original lines 188-201)
            if len(action_queue) == 0:
                with torch.no_grad():
                    # Prepare observation in VLA format
                    obs_dict = {
                        'rgb_obs': {
                            'rgb_static': torch.tensor(obs['rgb_static'] if isinstance(obs, dict) and 'rgb_static' in obs
                                                     else obs).unsqueeze(0).to(self.device),
                        },
                        'lang_text': [default_instruction]
                    }

                    # Add gripper cam if available
                    if isinstance(obs, dict) and 'rgb_gripper' in obs:
                        obs_dict['rgb_obs']['rgb_gripper'] = torch.tensor(obs['rgb_gripper']).unsqueeze(0).to(self.device)

                    # Use original VLA interface to sample action chunk
                    self.rl_vla.modality_scope = "lang"
                    obs_features = self.rl_vla.encode_observations(obs_dict)

                    # Sample action chunk (not single action like original, but chunk)
                    noise = torch.randn(
                        1, self.q_cfg.chunk_size, self.q_networks.action_dim,
                        device=self.device
                    )
                    action_chunk = self.rl_vla.sample_actions(noise, obs_features, inference=True)
                    action_chunk = action_chunk.cpu().numpy().reshape(-1, self.q_networks.action_dim)

                    # Add chunk to queue
                    for action in action_chunk:
                        action_queue.append(action)

            # Execute single action from queue (like original line 202)
            action = action_queue.pop(0)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Format observations for replay buffer (matching original VLA dataloader format)
            current_obs = {
                'rgb_obs': {
                    'rgb_static': obs['rgb_static'] if isinstance(obs, dict) else obs
                },
                'lang_text': default_instruction
            }

            next_obs_formatted = {
                'rgb_obs': {
                    'rgb_static': next_obs['rgb_static'] if isinstance(next_obs, dict) else next_obs
                },
                'lang_text': default_instruction
            }

            # Add other modalities if available
            if isinstance(obs, dict):
                if 'rgb_gripper' in obs:
                    current_obs['rgb_obs']['rgb_gripper'] = obs['rgb_gripper']
                if 'robot_obs' in obs:
                    current_obs['robot_obs'] = obs['robot_obs']

            if isinstance(next_obs, dict):
                if 'rgb_gripper' in next_obs:
                    next_obs_formatted['rgb_obs']['rgb_gripper'] = next_obs['rgb_gripper']
                if 'robot_obs' in next_obs:
                    next_obs_formatted['robot_obs'] = next_obs['robot_obs']

            # Create transition in original Q-chunking format (line 237-244)
            transition = {
                'observations': current_obs,
                'actions': action.copy(),
                'rewards': reward,
                'terminals': float(done),
                'masks': 1.0 - float(terminated),  # mask out terminated states
                'next_observations': next_obs_formatted,
            }

            # Add individual transition to replay buffer (like original line 245)
            self.replay_buffer.add_transition(
                obs=transition['observations'],
                action=transition['actions'],
                reward=transition['rewards'],
                next_obs=transition['next_observations'],
                done=bool(transition['terminals'])
            )
            transitions_collected += 1

            # Update current observation
            obs = next_obs

            # Reset action queue when episode ends
            if done:
                action_queue = []

        logger.debug(f"Collected {transitions_collected} transitions in episode")
        return transitions_collected

    def initialize_replay_buffer_with_demos(self, demo_dataloader):
        """
        Initialize replay buffer with expert demonstrations.
        Following original Q-chunking: replay buffer starts with expert data.
        """
        logger.info("Pre-loading replay buffer with expert demonstrations...")

        # Sample demonstrations and add to replay buffer
        demo_count = 0
        for batch_data in demo_dataloader:
            if demo_count >= 1000:  # Limit initial demo loading
                break

            batch = batch_data if isinstance(batch_data, dict) else {'lang': batch_data}

            # Extract trajectory data for replay buffer
            for i in range(len(batch['actions'])):
                # Format single trajectory
                trajectory = {
                    'actions': batch['actions'][i].cpu().numpy(),
                    'rewards': batch.get('rewards', torch.ones(len(batch['actions'][i]))).cpu().numpy(),
                    'dones': batch.get('dones', torch.zeros(len(batch['actions'][i]))).cpu().numpy(),
                    'rgb_obs': batch['rgb_obs']['rgb_static'][i].cpu().numpy(),
                    'instruction': batch['lang_text'][i] if isinstance(batch['lang_text'], list) else batch['lang_text']
                }

                # Add optional modalities
                if 'rgb_gripper' in batch['rgb_obs']:
                    trajectory['rgb_gripper'] = batch['rgb_obs']['rgb_gripper'][i].cpu().numpy()
                if 'robot_obs' in batch:
                    trajectory['robot_obs'] = batch['robot_obs'][i].cpu().numpy()

                # Add trajectory to replay buffer
                self.replay_buffer.add_trajectory(trajectory)
                demo_count += 1

        logger.info(f"Loaded {demo_count} expert demonstrations into replay buffer")

    def collect_single_transition(self, obs, action_queue, episode_step, env):
        """
        Collect single transition following original Q-chunking approach.

        Returns updated (obs, action_queue, episode_step) for next call.
        """
        default_instruction = "manipulate object"

        # Sample action chunk if queue empty (main_online.py:188-201)
        if len(action_queue) == 0:
            with torch.no_grad():
                # Format observation
                obs_dict = {
                    'rgb_obs': {
                        'rgb_static': torch.tensor(obs['rgb_static'] if isinstance(obs, dict) and 'rgb_static' in obs
                                                 else obs).unsqueeze(0).to(self.device),
                    },
                    'lang_text': [default_instruction]
                }

                if isinstance(obs, dict) and 'rgb_gripper' in obs:
                    obs_dict['rgb_obs']['rgb_gripper'] = torch.tensor(obs['rgb_gripper']).unsqueeze(0).to(self.device)

                # Sample action chunk using offline-trained RL VLA
                self.rl_vla.modality_scope = "lang"
                obs_features = self.rl_vla.encode_observations(obs_dict)

                noise = torch.randn(1, self.q_cfg.chunk_size, self.q_networks.action_dim, device=self.device)
                action_chunk = self.rl_vla.sample_actions(noise, obs_features, inference=True)
                action_chunk = action_chunk.cpu().numpy().reshape(-1, self.q_networks.action_dim)

                # Fill action queue
                for action in action_chunk:
                    action_queue.append(action)

        # Execute single action (main_online.py:202-252)
        action = action_queue.pop(0)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Format transition for replay buffer
        current_obs = {
            'rgb_obs': {'rgb_static': obs['rgb_static'] if isinstance(obs, dict) else obs},
            'lang_text': default_instruction
        }
        next_obs_formatted = {
            'rgb_obs': {'rgb_static': next_obs['rgb_static'] if isinstance(next_obs, dict) else next_obs},
            'lang_text': default_instruction
        }

        # Add modalities if available
        if isinstance(obs, dict):
            if 'rgb_gripper' in obs:
                current_obs['rgb_obs']['rgb_gripper'] = obs['rgb_gripper']
            if 'robot_obs' in obs:
                current_obs['robot_obs'] = obs['robot_obs']

        if isinstance(next_obs, dict):
            if 'rgb_gripper' in next_obs:
                next_obs_formatted['rgb_obs']['rgb_gripper'] = next_obs['rgb_gripper']
            if 'robot_obs' in next_obs:
                next_obs_formatted['robot_obs'] = next_obs['robot_obs']

        # Add transition to replay buffer
        self.replay_buffer.add_transition(
            obs=current_obs,
            action=action,
            reward=reward,
            next_obs=next_obs_formatted,
            done=done
        )

        # Handle episode completion
        if done:
            next_obs = env.reset()
            action_queue = []
            episode_step = 0
        else:
            episode_step += 1

        return next_obs, action_queue, episode_step
def clear_cuda_cache():
    """Clear CUDA cache and garbage collect unused memory."""
    if torch.cuda.is_available():
        # Empty CUDA cache
        torch.cuda.empty_cache()
        # Force garbage collection
        import gc
        gc.collect()
        # Log memory stats
        for i in range(torch.cuda.device_count()):
            memory_stats = torch.cuda.memory_stats(i)
            allocated = memory_stats.get('allocated_bytes.all.current', 0) / (1024**3)
            reserved = memory_stats.get('reserved_bytes.all.current', 0) / (1024**3)
            logger.info(f"GPU {i} Memory: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def log_rank_0(*args, **kwargs):
    logger.info(*args, **kwargs)

def setup_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    return [hydra.utils.instantiate(cb) for cb in callbacks_cfg.values()]

def setup_logger(cfg: DictConfig, model: LightningModule):
    pathlib_cwd = Path.cwd()
    if "group" in cfg.logger:
        cfg.logger.group = pathlib_cwd.parent.name
        cfg.logger.name = f"{pathlib_cwd.parent.name}/{pathlib_cwd.name}"
        cfg.logger.id = cfg.logger.name.replace("/", "_")
    return hydra.utils.instantiate(cfg.logger)



def _train_q_chunking_offline_stage(cfg: DictConfig, datamodule):

    """
    Q-chunking offline stage: Train BC + RL using ONLY offline dataset.
    Both policies learn from demonstrations.
    Creates TWO separate VLA models: BC VLA + RL VLA
    """
    log_rank_0("Starting Q-chunking offline stage")

    # Initialize trainer with offline configuration (creates 2 models: BC + RL)
    device = torch.device(cfg.device)
    q_trainer = QChunkingTrainer(cfg, device, env=None)
    q_trainer.setup_models(cfg.model)  # This creates BC VLA (4 steps) + RL VLA (1 step)
    q_trainer.setup_optimizers()

    log_rank_0(f"Created BC VLA with {q_trainer.bc_vla.num_sampling_steps} sampling steps")
    log_rank_0(f"Created RL VLA with {q_trainer.rl_vla.num_sampling_steps} sampling steps")

    # Setup data
    datamodule.setup("fit")
    train_dataloader = datamodule.train_dataloader()['lang']

    # Save initial checkpoint before any training
    initial_checkpoint_path = Path.cwd() / "checkpoint_initial.pt"
    q_trainer.save_checkpoint(initial_checkpoint_path, cfg.checkpoint_type)
    log_rank_0("✅ Saved initial checkpoint before training")

    # Train for configured epochs
    for epoch in range(cfg.max_epochs):
        log_rank_0(f"Offline Epoch {epoch + 1}/{cfg.max_epochs}")

        epoch_metrics = {'bc_loss': 0.0,'q_value':0.0 ,'q_loss': 0.0, 'distill_loss': 0.0,'rl_actor_loss':0.0}

        #加上日志打印
        progress_bar=tqdm(train_dataloader,desc=f"Epoch {epoch + 1}/{cfg.max_epochs}")
        for batch_idx, dataset_batch in enumerate(progress_bar):
            # Format batch for VLA

            batch = {'lang': dataset_batch}
            batch = _move_batch_to_device(batch, q_trainer.device)

            # Offline training: BC + RL both learn from dataset
            step_metrics = _q_chunking_offline_step(q_trainer, batch)
            for key, value in step_metrics.items():
                epoch_metrics[key] += value

        # Average metrics and log to wandb
        for key in epoch_metrics:
            epoch_metrics[key] /= len(train_dataloader)

        log_rank_0(f"Offline epoch {epoch + 1} metrics: {epoch_metrics}")
        # Log to wandb
        wandb_metrics = {f"offline/{k}": v for k, v in epoch_metrics.items()}
        wandb_metrics["offline/epoch"] = epoch + 1
        wandb.log(wandb_metrics)

        # Save periodic checkpoints based on config save_freq
        if (epoch + 1) % cfg.q_chunking.checkpoint.save_freq == 0:
            checkpoint_path = Path.cwd() / f"checkpoint_epoch_{epoch+1}.pt"
            q_trainer.save_checkpoint(checkpoint_path, "full")
            log_rank_0(f"✅ Saved checkpoint at epoch {epoch+1}")

    # Save final checkpoint
    checkpoint_path = Path.cwd() / "checkpoint_offline_complete.pt"
    q_trainer.save_checkpoint(checkpoint_path, cfg.checkpoint_type)
    log_rank_0("Q-chunking offline stage completed")


def _train_q_chunking_online_stage(cfg: DictConfig, datamodule, env=None):
    """
    Q-chunking online stage: Train BC + RL using MIXED data (dataset + replay buffer).
    Uses TWO separate VLA models: BC VLA + RL VLA
    """
    log_rank_0("Starting Q-chunking online stage")
    # Initialize trainer (creates 2 models: BC + RL)
    device = torch.device(cfg.device)
    q_trainer = QChunkingTrainer(cfg,  device , env)
    q_trainer.setup_models(cfg.model)  # This creates BC VLA (4 steps) + RL VLA (1 step)
    q_trainer.setup_optimizers()

    log_rank_0(f"Created BC VLA with {q_trainer.bc_vla.num_sampling_steps} sampling steps")
    log_rank_0(f"Created RL VLA with {q_trainer.rl_vla.num_sampling_steps} sampling steps")

    # Load from offline stage if available

    offline_checkpoint = Path.cwd() / "checkpoint_offline_complete.pt"
    if offline_checkpoint.exists():

        q_trainer.load_checkpoint(offline_checkpoint, "full")
        log_rank_0("Loaded offline stage checkpoint")


    # Setup data
    datamodule.setup("fit")
    train_dataloader = datamodule.train_dataloader()['lang']
    initial_checkpoint_path = Path.cwd() / "checkpoint_initial.pt"
    q_trainer.save_checkpoint(initial_checkpoint_path, cfg.checkpoint_type)

    # Train for configured epochs
    for epoch in range(cfg.max_epochs):
        log_rank_0(f"Online Epoch {epoch + 1}/{cfg.max_epochs}")

        # Collect online transitions periodically
        if env is not None and epoch % 2 == 0:
            transitions_collected = q_trainer.collect_online_transitions()
            log_rank_0(f"Collected {transitions_collected} transitions")

        epoch_metrics = {'bc_loss': 0.0, 'q_value':0.0, 'q_loss': 0.0, 'distill_loss': 0.0,'rl_actor_loss':0.0}
        progress_bar=tqdm(train_dataloader,desc=f'Epoch {epoch+1}/{cfg.max_epochs}')


        for batch_idx, dataset_batch in enumerate(progress_bar):

            # Format batch for VLA

            batch = {'lang': dataset_batch}
            batch = _move_batch_to_device(batch, q_trainer.device)

            # Online training: BC + RL learn from mixed data
            step_metrics = _q_chunking_online_step(q_trainer, batch)

            for key, value in step_metrics.items():
                epoch_metrics[key] += value

        # Average metrics and log to wandb
        for key in epoch_metrics:
            epoch_metrics[key] /= len(train_dataloader)


        log_rank_0(f"Online epoch {epoch + 1} metrics: {epoch_metrics}")

        # Log to wandb
        wandb_metrics = {f"online/{k}": v for k, v in epoch_metrics.items()}
        wandb_metrics["online/epoch"] = epoch + 1
        wandb_metrics["online/replay_buffer_size"] = len(q_trainer.replay_buffer)
        wandb.log(wandb_metrics)

        if (epoch + 1) % cfg.q_chunking.checkpoint.save_freq == 0:
            checkpoint_path = Path.cwd() / f"checkpoint_epoch_{epoch+1}.pt"
            q_trainer.save_checkpoint(checkpoint_path, cfg.checkpoint_type)
            log_rank_0(f"✅ Saved checkpoint at epoch {epoch+1}")


    checkpoint_path = Path.cwd() / "checkpoint_online_complete.pt"
    q_trainer.save_checkpoint(checkpoint_path, cfg.checkpoint_type)
    log_rank_0("Q-chunking online stage completed")


def _q_chunking_offline_step(q_trainer, batch: Dict[str, Any]) -> Dict[str, float]:
    """Execute one offline training step: BC + RL learning from dataset only."""
    metrics = {}

    batch=batch['lang'] if isinstance(batch,dict) and 'lang' in batch else batch

     # 1. BC learns from dataset
    bc_loss = q_trainer.compute_bc_flow_matching_loss(batch, q_trainer.bc_vla)
    q_trainer.bc_optimizer.zero_grad()

    bc_loss.backward()
    q_trainer.bc_optimizer.step()

    metrics['bc_loss'] = bc_loss.item()

    # 2. Q-networks learn from dataset (for value estimation)
    #q_loss = q_trainer.compute_q_value_loss(batch['lang'])
    q_loss=q_trainer.compute_q_value_loss(batch)
    q_trainer.q_optimizer.zero_grad()

    q_loss.backward()
    q_trainer.q_optimizer.step()
    metrics['q_loss'] = q_loss.item()

    # 3. RL VLA learns: BC flow + distillation + Q maximization (ACFQL formula 13)
    q_trainer.rl_optimizer.zero_grad()

    # Distillation loss (RL VLA distills from BC VLA)
    distill_loss = q_trainer.compute_distillation_loss(batch)

    # Q maximization loss (RL VLA actions should get high Q-values)
    rl_actor_loss, q_value = q_trainer.compute_rl_actor_loss(batch)

    # Combined RL loss (matching ACFQL formula 13: bc_flow + alpha*distill + q_maximize)
    total_rl_loss = (
        q_trainer.q_cfg.distill_loss_weight * distill_loss +
        rl_actor_loss  # Q maximization (negative Q for minimization)
    )

    total_rl_loss.backward()
    q_trainer.rl_optimizer.step()

    metrics['distill_loss'] = distill_loss.item()
    metrics['rl_actor_loss'] = rl_actor_loss.item()
    metrics['q_value'] = q_value.item()

    # 4. Update target networks
    if q_trainer.step_count % q_trainer.q_cfg.target_update_freq == 0:
        q_trainer._soft_update_target_networks()

    q_trainer.offline_steps += 1
    q_trainer.step_count += 1

    return metrics



def _q_chunking_online_step(q_trainer, batch: Dict[str, Any]) -> Dict[str, float]:
    """Execute one online training step: BC + RL learning from mixed data."""
    metrics = {}
    batch=batch['lang'] if isinstance(batch,dict) and 'lang' in batch else batch

    mixed_batch = q_trainer.replay_buffer.sample_mixed_batch(
        demo_batch=batch,
        sequence_length=q_trainer.q_cfg.chunk_size,
        discount=q_trainer.q_cfg.gamma
    )
    # 1. BC learns from dataset (continues learning from demos)
    bc_loss = q_trainer.compute_bc_flow_matching_loss(mixed_batch, q_trainer.bc_vla)
    q_trainer.bc_optimizer.zero_grad()
    bc_loss.backward()
    q_trainer.bc_optimizer.step()
    metrics['bc_loss'] = bc_loss.item()


    # 2. RL VLA learns: BC flow + distillation + Q maximization (ACFQL formula 13)

    q_trainer.rl_optimizer.zero_grad()

    # Distillation loss (RL VLA distills from BC VLA)
    distill_loss = q_trainer.compute_distillation_loss(mixed_batch)

    # Q maximization loss (RL VLA actions should get high Q-values)
    rl_actor_loss, q_value = q_trainer.compute_rl_actor_loss(mixed_batch)

    # Combined RL loss (matching ACFQL formula 13)
    total_rl_loss = (


        q_trainer.q_cfg.distill_loss_weight * distill_loss +
        rl_actor_loss  # Q maximization
    )

    total_rl_loss.backward()
    q_trainer.rl_optimizer.step()

    q_trainer.q_optimizer.zero_grad()
    q_loss=q_trainer.compute_q_value_loss()
    q_loss.backward()
    q_trainer.q_optimizer.step()


    metrics['distill_loss'] = distill_loss.item()
    metrics['rl_actor_loss'] = rl_actor_loss.item()
    metrics['q_value'] = q_value.item()

    # 4. Update target networks
    if q_trainer.step_count % q_trainer.q_cfg.target_update_freq == 0:
        q_trainer._soft_update_target_networks()

    q_trainer.online_steps += 1
    q_trainer.step_count += 1

    return metrics


def train_q_chunking(cfg: DictConfig, datamodule):
    """Main Q-chunking training coordinator that routes to correct training stages."""
    log_rank_0(f"🚀 Starting Q-chunking training: {cfg.training_choice}")

    env = None

    # Initialize environment for online training if needed
    if cfg.training_choice in ["offline","online", "offline2online"]:
        logger.info("start")
        try:
            log_rank_0("Note: Environment setup would depend on LIBERO configuration")
        except Exception as e:
            log_rank_0(f"Warning: Could not initialize environment: {e}")


    if cfg.training_choice == "offline":
        log_rank_0("🔥 Starting Q-chunking OFFLINE stage")
        _train_q_chunking_offline_stage(cfg, datamodule)


    elif cfg.training_choice == "online":
        log_rank_0("🌐 Starting Q-chunking ONLINE stage")
        _train_q_chunking_online_stage(cfg, datamodule, env)


    elif cfg.training_choice == "offline2online":
        log_rank_0("🔄 Starting Q-chunking OFFLINE-TO-ONLINE sequential training")
        log_rank_0("Phase 1: Offline RL stage")
        _train_q_chunking_offline_stage(cfg, datamodule)
        log_rank_0("Phase 2: Online RL stage")
        _train_q_chunking_online_stage(cfg, datamodule, env)


    else:
        raise ValueError(f"Unknown training choice: {cfg.training_choice}")


def _move_batch_to_device(batch, device):
    """Move batch data to specified device."""
    for key, value in batch.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if torch.is_tensor(sub_value):
                    batch[key][sub_key] = sub_value.to(device)
        elif torch.is_tensor(value):
            batch[key] = value.to(device)
    return batch


@hydra.main(config_path="../conf", config_name="config_libero")
def train(cfg: DictConfig) -> None:
    try:
        # Setup environment
        os.environ['HYDRA_FULL_ERROR'] = '1'
        # Set memory allocation configuration
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        seed_everything(cfg.seed, workers=True)
        torch.set_float32_matmul_precision('medium')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Clear CUDA cache before initialization
        clear_cuda_cache()

        device = torch.device(cfg.device)

        # Initialize components
        log_rank_0(f"\nInitializing training for seed {cfg.seed}")
        datamodule = hydra.utils.instantiate(cfg.datamodule)

        # Route training based on training_choice: BC/offline/online/offline2online
        if hasattr(cfg, 'training_choice') and cfg.training_choice != "BC":
            log_rank_0(f"🚀 Q-Chunking RL training mode: {cfg.training_choice}")
            log_rank_0("Bypassing PyTorch Lightning - using custom Q-chunking training loop")

            # Get current timestamp for consistent directory naming
            timestamp = datetime.now().strftime("%H-%M-%S")
            # Create logs directory with timestamp and training choice
            base_path = Path.cwd()
            log_dir = base_path
            log_dir.mkdir(parents=True, exist_ok=True)

            # Create work directory with additional seed info
            work_dir = log_dir / f"seed_{cfg.seed}_{cfg.training_choice}"
            work_dir.mkdir(exist_ok=True)
            os.chdir(work_dir)

            # Initialize Q-chunking specific wandb logging (NOT PyTorch Lightning logger)

            wandb.init(
                project=cfg.q_chunking.logging.project,
                entity=cfg.q_chunking.logging.entity,
                group=cfg.q_chunking.logging.group.replace("${training_choice}", cfg.training_choice),
                name=f"qchunking_{cfg.training_choice}_seed_{cfg.seed}",
                config=OmegaConf.to_container(cfg, resolve=True),
                save_code=True,
                dir=str(log_dir)
            )

            log_rank_0(f" Logs will be saved to: {log_dir}")


            # Initialize environment for online training if needed
            env = None
            if cfg.training_choice in ["offline","online", "offline2online"]:
                try:
                    # For LIBERO environment, you would initialize it here
                    # This is a placeholder - actual environment initialization would depend on LIBERO setup
                    log_rank_0("Environment initialization needed for online training")
                    log_rank_0("Note: Environment setup would depend on LIBERO configuration")

                except Exception as e:
                    log_rank_0(f"Warning: Could not initialize environment: {e}")
                    log_rank_0("Continuing with dataset-only training")

                # Route to appropriate Q-chunking training method
            train_q_chunking(cfg, datamodule)


        elif cfg.training_choice == "BC":
            # Original Flower-VLA BC training path (no Q-chunking)
            log_rank_0("📚 BC MODE: Using original Flower-VLA imitation learning with PyTorch Lightning")

            # Get current timestamp for consistent directory naming
            timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
            # Create logs directory with timestamp and training choice
            base_path = Path.cwd()
            log_dir = base_path
            log_dir.mkdir(parents=True, exist_ok=True)


            # Create model using standard approach
            model = hydra.utils.instantiate(cfg.model) if get_last_checkpoint(Path.cwd()) is None else \
                   getattr(models_m, cfg.model["_target_"].split(".")[-1]).load_from_checkpoint(get_last_checkpoint(Path.cwd()).as_posix())


            if "pretrain_chk" in cfg:
                initialize_pretrained_weights(model, cfg)

            # Setup PyTorch Lightning training components (uses cfg.logger and cfg.trainer)
            train_logger = setup_logger(cfg, model)  # Uses cfg.logger for PyTorch Lightning WandbLogger
            callbacks = setup_callbacks(cfg.callbacks) + [LearningRateMonitor(logging_interval="step")]

            # Set unique working directory for each seed
            work_dir = log_dir / f"seed_{cfg.seed}_BC"
            work_dir.mkdir(exist_ok=True)
            os.chdir(work_dir)

            log_rank_0(f"💾 Logs will be saved to: {log_dir}")

            trainer_args = {
                **cfg.trainer,
                "logger": train_logger,
                "callbacks": callbacks,
                "benchmark": False,
                "strategy": "ddp_find_unused_parameters_true",
                "accelerator": "gpu",
                "devices": cfg.trainer.devices,
                "use_distributed_sampler": True,
                "default_root_dir": work_dir,
                "sync_batchnorm": True,
                'precision':32,
            }

            # Log configuration
            log_rank_0(f"BC training config for seed {cfg.seed}")
            log_rank_0(f"Git commit: {get_git_commit_hash(Path(hydra.utils.to_absolute_path(__file__)))}")
            log_rank_0(print_system_env_info())

            # Clear CUDA cache again before training
            clear_cuda_cache()

            # Initialize PyTorch Lightning trainer and train
            trainer = Trainer(**trainer_args)

            try:
                trainer.fit(model, datamodule=datamodule)
            except Exception as e:
                log_rank_0("\nDetailed Error Information:")
                log_rank_0("=" * 80)
                log_rank_0(f"Error Type: {type(e).__name__}")
                log_rank_0(f"Error Message: {str(e)}")
                log_rank_0("\nFull Traceback:")
                import traceback
                log_rank_0(''.join(traceback.format_tb(e.__traceback__)))
                log_rank_0("\nLocal Variables at Crash Point:")
                tb = e.__traceback__
                while tb.tb_next:
                    tb = tb.tb_next
                log_rank_0(f"{traceback.extract_tb(tb)}")
                log_rank_0("=" * 80)
                raise e

        else:
            # Invalid training choice
            raise ValueError(f"Invalid training_choice: {getattr(cfg, 'training_choice', 'MISSING')}. Must be one of: BC, offline, online, offline2online")

    except Exception as e:
        logger.error(f"\nTraining failed for seed {cfg.seed}:")
        logger.error(f"{'='*80}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full traceback:")
        import traceback
        logger.error(traceback.format_exc())
        logger.error(f"{'='*80}")
        raise e
    finally:
        # Clear CUDA cache one final time
        clear_cuda_cache()
        # Clean up
        cleanup_distributed()
        # Ensure wandb is initialized before accessing wandb.run
        if 'wandb' in globals() and wandb.run is not None:
            wandb.finish()

def cleanup_distributed():
    """Cleanup distributed training resources"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()




if __name__ == "__main__":
    # Set environment variables
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = 'True'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
    # Add repo to path
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

    try:
        train()
    except Exception as e:
        logger.error(f"\nTraining script failed:")
        logger.error(f"{'='*80}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full traceback:")
        import traceback
        logger.error(traceback.format_exc())
        logger.error(f"{'='*80}")
        sys.exit(1)