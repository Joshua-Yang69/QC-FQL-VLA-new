import logging
from pathlib import Path
import sys
sys.tracebacklimit = None
import os
import wandb
import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
import copy
import numpy as np
from typing import Dict, Any, Optional, Tuple


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

    def __init__(self, cfg: DictConfig, device: torch.device, env=None):
        self.cfg = cfg
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
        self.bc_vla = hydra.utils.instantiate(base_model_cfg).to(self.device)
        self.bc_vla.num_sampling_steps = 4

        # Load pretrained weights if available
        if "pretrain_chk" in self.cfg and self.cfg.pretrain_chk:
            from flower.utils.utils import initialize_pretrained_weights
            initialize_pretrained_weights(self.bc_vla, self.cfg)
            logger.info(f"Loaded pretrained weights for BC VLA from {self.cfg.pretrain_chk}")

        logger.info("Initialized BC VLA with 4 sampling steps")

        # 2. RL VLA (identical to BC VLA, used for Q-learning) with 1 sampling step for fast online training
        self.rl_vla = hydra.utils.instantiate(base_model_cfg).to(self.device)
        self.rl_vla.num_sampling_steps = 1
        logger.info("Initialized RL VLA with 1 sampling step")

        # 3. Double Q-networks for value estimation
        self.q_networks = DoubleQNetwork(
            state_dim=self.q_cfg.q_network.state_dim,
            action_dim=self.q_cfg.q_network.action_dim,
            chunk_size=self.q_cfg.chunk_size,
            hidden_dim=self.q_cfg.q_network.hidden_dim,
            n_layers=self.q_cfg.q_network.n_layers,
            dropout=self.q_cfg.q_network.dropout
        ).to(self.device)
        logger.info("Initialized Q-networks")

        # 4. Target Q-networks (copies for stable training)
        self.target_q_networks = DoubleQNetwork(
            state_dim=self.q_cfg.q_network.state_dim,
            action_dim=self.q_cfg.q_network.action_dim,
            chunk_size=self.q_cfg.chunk_size,
            hidden_dim=self.q_cfg.q_network.hidden_dim,
            n_layers=self.q_cfg.q_network.n_layers,
            dropout=self.q_cfg.q_network.dropout
        ).to(self.device)

        # Initialize target networks with main network weights
        self.target_q_networks.load_state_dict(self.q_networks.state_dict())
        logger.info("Initialized target Q-networks")

        # 5. Setup replay buffer
        self.replay_buffer = SequenceReplayBuffer(
            capacity=self.q_cfg.replay_buffer_size,
            chunk_size=self.q_cfg.chunk_size,
            device=self.device
        )
        logger.info("Initialized replay buffer")

    def setup_optimizers(self):
        """Setup optimizers for all models."""
        # BC VLA optimizer
        self.bc_optimizer = optim.AdamW(
            self.bc_vla.parameters(),
            lr=self.cfg.model.optimizer.learning_rate,
            betas=self.cfg.model.optimizer.betas,
            weight_decay=self.cfg.model.optimizer.transformer_weight_decay
        )

        # RL VLA optimizer (same as BC)
        self.rl_optimizer = optim.AdamW(
            self.rl_vla.parameters(),
            lr=self.cfg.model.optimizer.learning_rate,
            betas=self.cfg.model.optimizer.betas,
            weight_decay=self.cfg.model.optimizer.transformer_weight_decay
        )

        # Q-networks optimizer
        self.q_optimizer = optim.AdamW(
            self.q_networks.parameters(),
            lr=self.q_cfg.q_lr,
            betas=self.cfg.model.optimizer.betas,
            weight_decay=0.01
        )

        logger.info("Setup optimizers")

    def compute_bc_flow_matching_loss(self, batch: Dict[str, Any], model) -> torch.Tensor:
        """Compute behavior cloning flow matching loss using original VLA interface."""
        total_loss = 0.0
        num_modalities = 0
        
        # Handle original dataloader format: {"lang": {dataset_batch}}
        if isinstance(batch, dict) and any(isinstance(v, dict) and 'actions' in v for v in batch.values()):
            # Original batch format from dataloader
            for modality_scope, dataset_batch in batch.items():
                if isinstance(dataset_batch, dict) and 'actions' in dataset_batch:
                    # Use original VLA interface exactly as designed
                    model.modality_scope = modality_scope
                    obs_features = model.encode_observations(dataset_batch)
                    loss, _ = model.rf_loss(obs_features, dataset_batch["actions"])
                    total_loss += loss
                    num_modalities += 1
        else:
            # Direct dataset_batch format (for replay buffer data)
            # Need to wrap in expected format
            if 'actions' in batch:
                dataset_batch = batch
            else:
                raise ValueError(f"Invalid batch format - no actions found in {list(batch.keys())}")
                
            # Use original VLA interface
            model.modality_scope = "lang"  # Default modality
            obs_features = model.encode_observations(dataset_batch)
            loss, _ = model.rf_loss(obs_features, dataset_batch["actions"])
            total_loss = loss
            num_modalities = 1
        
        return total_loss / max(num_modalities, 1)

    def compute_q_value_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute Q-value loss for Q-learning using original VLA interface."""
        # Extract basic data from batch
        actions = batch['actions'].to(self.device)  
        rewards = batch['rewards'].to(self.device)  
        terminals = batch.get('terminals', batch.get('dones', torch.zeros(len(actions), dtype=torch.bool))).to(self.device)
        
        # Handle terminals format
        if terminals.dim() > 1:
            terminals = terminals.squeeze(-1)
        
        # Prepare current observation batch in original VLA format
        current_obs_batch = {
            'rgb_obs': batch['rgb_obs'] if 'rgb_obs' in batch else {'rgb_static': batch.get('observations', torch.zeros(1))},
            'lang_text': batch.get('instruction', ['manipulate object'] * len(actions))
        }
        
        # Add other observation modalities if available
        if 'robot_obs' in batch:
            current_obs_batch['robot_obs'] = batch['robot_obs']
        if 'proprio' in batch:
            current_obs_batch['robot_obs'] = batch['proprio']
            
        # Prepare next observation batch
        next_obs_batch = {
            'rgb_obs': batch.get('next_rgb_obs', current_obs_batch['rgb_obs']),
            'lang_text': batch.get('next_instruction', current_obs_batch['lang_text'])
        }
        
        if 'next_robot_obs' in batch:
            next_obs_batch['robot_obs'] = batch['next_robot_obs']
        elif 'robot_obs' in current_obs_batch:
            next_obs_batch['robot_obs'] = current_obs_batch['robot_obs']

        # Encode states using original VLA interface
        with torch.no_grad():
            # Set modality scope for RL VLA
            self.rl_vla.modality_scope = "lang"
            
            # Encode current states
            current_obs_features = self.rl_vla.encode_observations(current_obs_batch)
            current_states = current_obs_features['features'].mean(dim=1)  # Pool sequence dimension
            
            # Encode next states  
            next_obs_features = self.rl_vla.encode_observations(next_obs_batch)
            next_states = next_obs_features['features'].mean(dim=1)  # Pool sequence dimension

        # Ensure actions have correct shape for Q-networks
        if actions.dim() == 2:  # (B, action_dim) -> (B, 1, action_dim) 
            actions = actions.unsqueeze(1)
        
        # Current Q-values
        q1_current, q2_current = self.q_networks(current_states, actions)

        # Target Q-values with target networks
        with torch.no_grad():
            # Sample next actions using original VLA interface
            next_noise = torch.randn_like(actions, device=self.device)
            next_actions = self.rl_vla.sample_actions(next_noise, next_obs_features, inference=True)
            
            # Ensure next_actions has correct shape for Q-networks
            if next_actions.dim() == 2:
                next_actions = next_actions.unsqueeze(1)

            # Target Q-values
            target_q1, target_q2 = self.target_q_networks(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            # Compute targets - handle chunk rewards properly
            if rewards.dim() == 1:  # (B,) -> (B, 1)
                rewards = rewards.unsqueeze(1)
                
            if rewards.shape[1] == 1:
                # Single-step rewards, use directly
                chunk_returns = rewards
            else:
                # Multi-step rewards, compute discounted sum
                chunk_returns = torch.zeros_like(target_q)
                for i in range(min(rewards.shape[1], self.q_cfg.chunk_size)):
                    chunk_returns += rewards[:, i:i+1] * (self.q_cfg.discount_factor ** i)

            # Add discounted next state value if not terminal
            targets = chunk_returns + (1 - terminals.float().unsqueeze(1)) * (self.q_cfg.discount_factor ** self.q_cfg.chunk_size) * target_q

        # Compute Q-losses
        q1_loss = F.mse_loss(q1_current, targets)
        q2_loss = F.mse_loss(q2_current, targets)
        q_loss = q1_loss + q2_loss

        return q_loss

    def compute_distillation_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute distillation loss between BC and RL policies using original VLA interface."""
        total_distill_loss = 0.0
        num_samples = 0
        
        # Handle batch format properly
        if isinstance(batch, dict) and any(isinstance(v, dict) and 'actions' in v for v in batch.values()):
            # Original dataloader format: {"lang": {dataset_batch}}
            for modality_scope, dataset_batch in batch.items():
                if isinstance(dataset_batch, dict) and 'actions' in dataset_batch:
                    # Get batch size from actions
                    batch_size = len(dataset_batch["actions"])
                    
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
                    
                    # Compute distillation loss
                    distill_loss = F.mse_loss(rl_actions, bc_actions.detach())
                    total_distill_loss += distill_loss
                    num_samples += 1
        else:
            # Direct dataset_batch format (for replay buffer)
            if 'actions' in batch:
                dataset_batch = batch
                batch_size = len(dataset_batch["actions"])
                
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
                
                distill_loss = F.mse_loss(rl_actions, bc_actions.detach())
                total_distill_loss = distill_loss
                num_samples = 1
            else:
                raise ValueError(f"Invalid batch format - no actions found in {list(batch.keys())}")
        
        return total_distill_loss / max(num_samples, 1)

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute one training step."""
        metrics = {}

        # 1. BC Flow Matching Loss
        self.bc_optimizer.zero_grad()
        bc_loss = self.compute_bc_flow_matching_loss(batch, self.bc_vla)
        bc_loss.backward()
        self.bc_optimizer.step()
        metrics['bc_loss'] = bc_loss.item()

        # 2. Q-Value Loss (if replay buffer has enough data)
        if len(self.replay_buffer) >= self.q_cfg.batch_size:
            self.q_optimizer.zero_grad()
            replay_batch = self.replay_buffer.sample_sequence(
                batch_size=self.q_cfg.batch_size,
                discount=self.q_cfg.discount_factor
            )
            q_loss = self.compute_q_value_loss(replay_batch)
            q_loss.backward()
            self.q_optimizer.step()
            metrics['q_loss'] = q_loss.item()
        else:
            metrics['q_loss'] = 0.0

        # 3. RL Policy Update with Distillation Loss
        self.rl_optimizer.zero_grad()

        # RL flow matching loss (same as BC)
        rl_bc_loss = self.compute_bc_flow_matching_loss(batch, self.rl_vla)

        # Distillation loss
        distill_loss = self.compute_distillation_loss(batch)

        # Combined RL loss
        total_rl_loss = (
            rl_bc_loss +
            self.q_cfg.bc_distill_loss_weight * distill_loss
        )

        total_rl_loss.backward()
        self.rl_optimizer.step()

        metrics['rl_bc_loss'] = rl_bc_loss.item()
        metrics['distill_loss'] = distill_loss.item()

        # 4. Update target networks
        if self.step_count % self.q_cfg.target_update_freq == 0:
            self._soft_update_target_networks()

        self.step_count += 1
        return metrics

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

        Args:
            checkpoint_path: Path to save checkpoint
            checkpoint_type: Type of checkpoint - 'bc_only', 'rl_only', 'full'
        """
        checkpoint_data = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'step_count': self.step_count,
            'training_stage': self.training_stage,
            'offline_steps': self.offline_steps,
            'online_steps': self.online_steps,
            'config': self.cfg
        }

        if checkpoint_type in ["bc_only", "full"]:
            checkpoint_data.update({
                'bc_vla_state_dict': self.bc_vla.state_dict(),
                'bc_optimizer_state_dict': self.bc_optimizer.state_dict(),
            })

        if checkpoint_type in ["rl_only", "full"]:
            checkpoint_data.update({
                'rl_vla_state_dict': self.rl_vla.state_dict(),
                'rl_optimizer_state_dict': self.rl_optimizer.state_dict(),
            })

        if checkpoint_type == "full":
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
        """Save models in HuggingFace format using existing save_to_hf functions."""
        try:
            from flower.utils.save_to_hf import save_config, create_model_card
            from safetensors.torch import save_file
            import json

            base_dir = checkpoint_path.parent / f"hf_models_{checkpoint_path.stem}"

            if checkpoint_type in ["bc_only", "full"]:
                # Save BC VLA using existing pattern
                bc_dir = base_dir / "bc_vla"
                bc_dir.mkdir(parents=True, exist_ok=True)

                # Use existing save pattern from save_to_hf.py
                torch.save({'state_dict': self.bc_vla.state_dict()}, bc_dir / "model.pt")
                save_file(self.bc_vla.state_dict(), bc_dir / "model.safetensors")

                # Create minimal config using existing save_config pattern
                model_config = {
                    "model_type": "flower_vla_bc",
                    "num_sampling_steps": self.bc_vla.num_sampling_steps
                }
                with open(bc_dir / "config.json", 'w') as f:
                    json.dump(model_config, f, indent=2)

                create_model_card(bc_dir.as_posix())

            if checkpoint_type in ["rl_only", "full"]:
                # Save RL VLA using existing pattern
                rl_dir = base_dir / "rl_vla"
                rl_dir.mkdir(parents=True, exist_ok=True)

                torch.save({'state_dict': self.rl_vla.state_dict()}, rl_dir / "model.pt")
                save_file(self.rl_vla.state_dict(), rl_dir / "model.safetensors")

                model_config = {
                    "model_type": "flower_vla_rl",
                    "num_sampling_steps": self.rl_vla.num_sampling_steps
                }
                with open(rl_dir / "config.json", 'w') as f:
                    json.dump(model_config, f, indent=2)

                create_model_card(rl_dir.as_posix())

            if checkpoint_type == "full":
                # Save Q-networks using existing pattern
                q_dir = base_dir / "q_networks"
                q_dir.mkdir(parents=True, exist_ok=True)

                torch.save({'state_dict': self.q_networks.state_dict()}, q_dir / "model.pt")
                save_file(self.q_networks.state_dict(), q_dir / "model.safetensors")

                model_config = {"model_type": "q_networks"}
                with open(q_dir / "config.json", 'w') as f:
                    json.dump(model_config, f, indent=2)

                create_model_card(q_dir.as_posix())

            logger.info(f"Saved models in HuggingFace format: {base_dir}")

        except ImportError as e:
            logger.warning(f"Missing dependencies for HuggingFace format: {e}")
        except Exception as e:
            logger.warning(f"Failed to save HuggingFace format: {e}")

    def load_checkpoint(self, checkpoint_path: Path, load_type: str = "full"):
        """
        Load training checkpoint with specified type.

        Args:
            checkpoint_path: Path to load checkpoint from
            load_type: Type of checkpoint to load - 'bc_only', 'rl_only', 'full'
        """
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return False

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load training state
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.step_count = checkpoint.get('step_count', 0)
        self.training_stage = checkpoint.get('training_stage', self.training_stage)
        self.offline_steps = checkpoint.get('offline_steps', 0)
        self.online_steps = checkpoint.get('online_steps', 0)

        if load_type in ["bc_only", "full"] and 'bc_vla_state_dict' in checkpoint:
            self.bc_vla.load_state_dict(checkpoint['bc_vla_state_dict'])
            self.bc_optimizer.load_state_dict(checkpoint['bc_optimizer_state_dict'])
            logger.info("Loaded BC VLA checkpoint")

        if load_type in ["rl_only", "full"] and 'rl_vla_state_dict' in checkpoint:
            self.rl_vla.load_state_dict(checkpoint['rl_vla_state_dict'])
            self.rl_optimizer.load_state_dict(checkpoint['rl_optimizer_state_dict'])
            logger.info("Loaded RL VLA checkpoint")

        if load_type == "full":
            if 'q_networks_state_dict' in checkpoint:
                self.q_networks.load_state_dict(checkpoint['q_networks_state_dict'])
                self.target_q_networks.load_state_dict(checkpoint['target_q_networks_state_dict'])
                self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
                logger.info("Loaded Q-networks checkpoint")

            if 'replay_buffer_state' in checkpoint and checkpoint['replay_buffer_state']:
                self.replay_buffer.load_state(checkpoint['replay_buffer_state'])
                logger.info("Loaded replay buffer state")

        logger.info(f"Loaded {load_type} checkpoint from: {checkpoint_path}")
        return True

    def offline_train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute offline training step (BC only, following original Q-chunking algorithm)."""
        metrics = {}

        # Only BC Flow Matching Loss in offline stage
        self.bc_optimizer.zero_grad()
        bc_loss = self.compute_bc_flow_matching_loss(batch, self.bc_vla)
        bc_loss.backward()
        self.bc_optimizer.step()
        metrics['bc_loss'] = bc_loss.item()

        # Update RL VLA to match BC VLA (copy weights for initialization)
        if self.offline_steps % 1000 == 0:  # Sync every 1000 steps
            self.rl_vla.load_state_dict(self.bc_vla.state_dict())
            metrics['rl_sync'] = 1.0
        else:
            metrics['rl_sync'] = 0.0

        self.offline_steps += 1
        self.step_count += 1
        return metrics

    def online_train_step(self, batch: Dict[str, Any], replay_batch: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Execute online training step with environment interaction (following original Q-chunking algorithm)."""
        metrics = {}

        # 1. BC Flow Matching Loss (from dataset)
        self.bc_optimizer.zero_grad()
        bc_loss = self.compute_bc_flow_matching_loss(batch, self.bc_vla)
        bc_loss.backward()
        self.bc_optimizer.step()
        metrics['bc_loss'] = bc_loss.item()

        # 2. Q-Value Loss (from replay buffer if available)
        if replay_batch is not None and len(self.replay_buffer) >= self.q_cfg.batch_size:
            self.q_optimizer.zero_grad()
            q_loss = self.compute_q_value_loss(replay_batch)
            q_loss.backward()
            self.q_optimizer.step()
            metrics['q_loss'] = q_loss.item()
        else:
            metrics['q_loss'] = 0.0

        # 3. RL Policy Update with Distillation Loss (mixed dataset + replay data)
        self.rl_optimizer.zero_grad()

        # RL flow matching loss (same as BC)
        rl_bc_loss = self.compute_bc_flow_matching_loss(batch, self.rl_vla)

        # Distillation loss
        distill_loss = self.compute_distillation_loss(batch)

        # Combined RL loss
        total_rl_loss = (
            rl_bc_loss +
            self.q_cfg.bc_distill_loss_weight * distill_loss
        )

        total_rl_loss.backward()
        self.rl_optimizer.step()

        metrics['rl_bc_loss'] = rl_bc_loss.item()
        metrics['distill_loss'] = distill_loss.item()

        # 4. Update target networks
        if self.step_count % self.q_cfg.target_update_freq == 0:
            self._soft_update_target_networks()

        self.online_steps += 1
        self.step_count += 1
        return metrics

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
                        1, self.q_cfg.chunk_size, self.q_cfg.q_network.action_dim,
                        device=self.device
                    )
                    action_chunk = self.rl_vla.sample_actions(noise, obs_features, inference=True)
                    action_chunk = action_chunk.cpu().numpy().reshape(-1, self.q_cfg.q_network.action_dim)
                    
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

@rank_zero_only
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

def train_q_chunking(cfg: DictConfig, datamodule, device: torch.device, env=None):
    """
    Q-Chunking training loop implementation following original algorithm structure.
    Supports three modes: offline, online, offline-to-online
    """
    training_stage = cfg.q_chunking.training_stage
    log_rank_0(f"Starting Q-Chunking training in {training_stage} mode...")

    # Initialize Q-Chunking trainer
    q_trainer = QChunkingTrainer(cfg, device, env)

    # Setup models
    q_trainer.setup_models(cfg.model)
    q_trainer.setup_optimizers()

    # Load checkpoint if specified
    checkpoint_path = Path.cwd() / "checkpoint_latest.pt"
    if checkpoint_path.exists():
        if training_stage == "offline":
            q_trainer.load_checkpoint(checkpoint_path, "bc_only")
        elif training_stage == "online":
            q_trainer.load_checkpoint(checkpoint_path, "full")
        else:  # offline-to-online
            q_trainer.load_checkpoint(checkpoint_path, "full")
        log_rank_0(f"Loaded checkpoint for {training_stage} training")

    # Initialize data
    datamodule.setup("fit")
    train_dataloader = datamodule.train_dataloader()

    log_rank_0(f"Training with {len(train_dataloader)} batches per epoch")

    # Training based on stage
    if training_stage == "offline":
        _train_offline_stage(q_trainer, train_dataloader, cfg)
    elif training_stage == "online":
        _train_online_stage(q_trainer, train_dataloader, cfg, env)
    elif training_stage == "offline-to-online":
        # First offline stage
        log_rank_0("Phase 1: Offline training")
        _train_offline_stage(q_trainer, train_dataloader, cfg, switch_to_online_after=cfg.max_epochs//2)

        # Then online stage
        log_rank_0("Phase 2: Online training")
        q_trainer.training_stage = "online"
        _train_online_stage(q_trainer, train_dataloader, cfg, env, start_epoch=cfg.max_epochs//2)
    else:
        raise ValueError(f"Unknown training stage: {training_stage}")

    log_rank_0("Q-Chunking training completed!")

def _train_offline_stage(q_trainer, train_dataloader, cfg, switch_to_online_after=None):
    """
    Train offline stage (BC only) - following original Q-chunking main.py lines 174-214
    """
    max_epochs = switch_to_online_after if switch_to_online_after else cfg.max_epochs

    for epoch in range(q_trainer.epoch, max_epochs):
        log_rank_0(f"Offline Epoch {epoch + 1}/{max_epochs}")
        q_trainer.epoch = epoch

        epoch_metrics = {'bc_loss': 0.0, 'rl_sync': 0.0}

        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = _move_batch_to_device(batch, q_trainer.device)

            # Offline training step (BC only)
            step_metrics = q_trainer.offline_train_step(batch)

            # Update epoch metrics
            for key, value in step_metrics.items():
                epoch_metrics[key] += value

            # Log step metrics
            if q_trainer.global_step % 100 == 0:
                log_metrics = {f"offline_step/{k}": v for k, v in step_metrics.items()}
                log_metrics["offline_step/global_step"] = q_trainer.global_step
                log_metrics["offline_step/epoch"] = epoch
                log_metrics["offline_step/training_stage"] = "offline"
                wandb.log(log_metrics, step=q_trainer.global_step)

            q_trainer.global_step += 1

        # Log epoch metrics
        for key, value in epoch_metrics.items():
            epoch_metrics[key] = value / len(train_dataloader)

        log_rank_0(f"Offline Epoch {epoch + 1} metrics: {epoch_metrics}")

        epoch_log_metrics = {f"offline_epoch/{k}": v for k, v in epoch_metrics.items()}
        epoch_log_metrics["offline_epoch/epoch"] = epoch
        epoch_log_metrics["offline_epoch/training_stage"] = "offline"
        epoch_log_metrics["offline_epoch/offline_steps"] = q_trainer.offline_steps
        wandb.log(epoch_log_metrics, step=q_trainer.global_step)

        # Save checkpoints
        if (epoch + 1) % cfg.q_chunking.checkpoint.save_freq == 0:
            # Save BC checkpoint
            bc_checkpoint_path = Path.cwd() / f"bc_checkpoint_epoch_{epoch + 1}.pt"
            q_trainer.save_checkpoint(bc_checkpoint_path, "bc_only")

            # Save full checkpoint for potential online continuation
            full_checkpoint_path = Path.cwd() / f"full_checkpoint_epoch_{epoch + 1}.pt"
            q_trainer.save_checkpoint(full_checkpoint_path, "full")

            # Save latest checkpoint
            latest_checkpoint_path = Path.cwd() / "checkpoint_latest.pt"
            q_trainer.save_checkpoint(latest_checkpoint_path, "full")

def _train_online_stage(q_trainer, train_dataloader, cfg, env, start_epoch=0):
    """
    Train online stage (BC + RL + Q-learning) - following original Q-chunking main.py lines 231-326
    """
    if env is None:
        log_rank_0("Warning: No environment provided for online training, using dataset only")

    for epoch in range(start_epoch, cfg.max_epochs):
        log_rank_0(f"Online Epoch {epoch + 1}/{cfg.max_epochs}")
        q_trainer.epoch = epoch

        epoch_metrics = {
            'bc_loss': 0.0,
            'q_loss': 0.0,
            'rl_bc_loss': 0.0,
            'distill_loss': 0.0,
            'trajectories_collected': 0
        }

        # Collect online transitions periodically
        if env is not None and epoch % 2 == 0:  # Collect every 2 epochs  
            transitions_collected = q_trainer.collect_online_transitions()
            if transitions_collected > 0:
                epoch_metrics['trajectories_collected'] = 1
                epoch_metrics['transitions_collected'] = transitions_collected
                log_rank_0(f"Collected {transitions_collected} transitions in episode")

        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = _move_batch_to_device(batch, q_trainer.device)

            # Sample replay batch if buffer has enough data
            replay_batch = None
            if len(q_trainer.replay_buffer) >= cfg.q_chunking.batch_size:
                replay_batch = q_trainer.replay_buffer.sample_sequence(
                    batch_size=cfg.q_chunking.batch_size,
                    discount=cfg.q_chunking.discount_factor
                )

            # Online training step (BC + RL + Q-learning)
            step_metrics = q_trainer.online_train_step(batch, replay_batch)

            # Update epoch metrics
            for key, value in step_metrics.items():
                epoch_metrics[key] += value

            # Log step metrics
            if q_trainer.global_step % 100 == 0:
                log_metrics = {f"online_step/{k}": v for k, v in step_metrics.items()}
                log_metrics["online_step/global_step"] = q_trainer.global_step
                log_metrics["online_step/epoch"] = epoch
                log_metrics["online_step/training_stage"] = "online"
                log_metrics["online_step/replay_buffer_size"] = len(q_trainer.replay_buffer)
                log_metrics["online_step/online_steps"] = q_trainer.online_steps
                wandb.log(log_metrics, step=q_trainer.global_step)

            q_trainer.global_step += 1

        # Log epoch metrics
        for key, value in epoch_metrics.items():
            if key != 'trajectories_collected':
                epoch_metrics[key] = value / len(train_dataloader)

        log_rank_0(f"Online Epoch {epoch + 1} metrics: {epoch_metrics}")

        epoch_log_metrics = {f"online_epoch/{k}": v for k, v in epoch_metrics.items()}
        epoch_log_metrics["online_epoch/epoch"] = epoch
        epoch_log_metrics["online_epoch/training_stage"] = "online"
        epoch_log_metrics["online_epoch/replay_buffer_size"] = len(q_trainer.replay_buffer)
        epoch_log_metrics["online_epoch/online_steps"] = q_trainer.online_steps
        wandb.log(epoch_log_metrics, step=q_trainer.global_step)

        # Save checkpoints
        if (epoch + 1) % cfg.q_chunking.checkpoint.save_freq == 0:
            # Save separate checkpoints
            bc_checkpoint_path = Path.cwd() / f"bc_checkpoint_epoch_{epoch + 1}.pt"
            q_trainer.save_checkpoint(bc_checkpoint_path, "bc_only")

            rl_checkpoint_path = Path.cwd() / f"rl_checkpoint_epoch_{epoch + 1}.pt"
            q_trainer.save_checkpoint(rl_checkpoint_path, "rl_only")

            full_checkpoint_path = Path.cwd() / f"full_checkpoint_epoch_{epoch + 1}.pt"
            q_trainer.save_checkpoint(full_checkpoint_path, "full")

            # Save latest checkpoint
            latest_checkpoint_path = Path.cwd() / "checkpoint_latest.pt"
            q_trainer.save_checkpoint(latest_checkpoint_path, "full")

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

        # Check if Q-chunking is enabled
        if hasattr(cfg, 'q_chunking') and cfg.q_chunking.enabled:
            log_rank_0(f"Q-Chunking enabled - using Q-chunking training in {cfg.q_chunking.training_stage} mode")

            # Setup logger for Q-chunking
            pathlib_cwd = Path.cwd()
            if "group" in cfg.logger:
                cfg.logger.group = pathlib_cwd.parent.name + "_qchunking_" + cfg.q_chunking.training_stage
                cfg.logger.name = f"{pathlib_cwd.parent.name}_qchunking_{cfg.q_chunking.training_stage}/{pathlib_cwd.name}"
                cfg.logger.id = cfg.logger.name.replace("/", "_")
            train_logger = setup_logger(cfg, None)

            # Set unique working directory for each seed and training stage
            work_dir = Path.cwd() / f"seed_{cfg.seed}_qchunking_{cfg.q_chunking.training_stage}"
            work_dir.mkdir(exist_ok=True)
            os.chdir(work_dir)

            # Initialize environment for online training if needed
            env = None
            if cfg.q_chunking.training_stage in ["online", "offline-to-online"]:
                try:
                    # For LIBERO environment, you would initialize it here
                    # This is a placeholder - actual environment initialization would depend on LIBERO setup
                    log_rank_0("Environment initialization needed for online training")
                    log_rank_0("Note: Environment setup would depend on LIBERO configuration")
                except Exception as e:
                    log_rank_0(f"Warning: Could not initialize environment: {e}")
                    log_rank_0("Continuing with dataset-only training")

            # Start Q-chunking training
            train_q_chunking(cfg, datamodule, device, env)

        else:
            # Original PyTorch Lightning training
            log_rank_0("Using original PyTorch Lightning training")
            model = hydra.utils.instantiate(cfg.model) if get_last_checkpoint(Path.cwd()) is None else \
                   getattr(models_m, cfg.model["_target_"].split(".")[-1]).load_from_checkpoint(get_last_checkpoint(Path.cwd()).as_posix())

            if "pretrain_chk" in cfg:
                initialize_pretrained_weights(model, cfg)

            # Setup training
            train_logger = setup_logger(cfg, model)
            callbacks = setup_callbacks(cfg.callbacks) + [LearningRateMonitor(logging_interval="step")]

            # Set unique working directory for each seed
            work_dir = Path.cwd() / f"seed_{cfg.seed}"
            work_dir.mkdir(exist_ok=True)
            os.chdir(work_dir)

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
            }

            # Log configuration
            log_rank_0(f"Training config for seed {cfg.seed}:\n{cfg}")
            log_rank_0(f"Git commit: {get_git_commit_hash(Path(hydra.utils.to_absolute_path(__file__)))}")
            log_rank_0(print_system_env_info())

            # Clear CUDA cache again before training
            clear_cuda_cache()

            # Initialize trainer and train
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
        if wandb.run is not None:
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