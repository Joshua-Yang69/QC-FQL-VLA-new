"""
Replay Buffer for Online RL
===========================

Sequence-aware replay buffer implementation for Q-chunking with action chunks.
Based on the original Q-chunking ACFQL implementation.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union
from collections import deque
import random
import logging

logger = logging.getLogger(__name__)


class SequenceReplayBuffer:
    """
    Sequence-aware replay buffer for Q-chunking with action chunks.

    Based on the original Q-chunking ACFQL implementation:
    - Stores complete trajectories rather than single transitions
    - Supports sequence sampling with proper discount computation
    - Handles terminal states correctly across chunk boundaries
    - Supports initialization from offline dataset
    """

    def __init__(self,
                 capacity: int = 100000,
                 chunk_size: int = 10,
                 device: Optional[torch.device] = None,
                 store_on_gpu: bool = False):
        """
        Initialize sequence-aware replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            chunk_size: Action chunk size for sequence sampling
            device: Device for sampled batches
            store_on_gpu: If True, store all data on GPU (faster but uses more memory)
        """
        self.capacity = capacity
        self.chunk_size = chunk_size
        self.device = device or torch.device('cpu')
        self.store_on_gpu = store_on_gpu and self.device.type == 'cuda'

        # Use numpy arrays for efficient storage and indexing
        self.size = 0
        self.ptr = 0
        self.initialized = False

        # Storage arrays (will be initialized when first data is added)
        self.observations = None
        self.actions = None
        self.rewards = None
        self.terminals = None
        self.next_observations = None
        self.episode_starts = []  # Track episode boundaries

        logger.info(f"Created sequence-aware replay buffer with capacity {capacity}, chunk_size {chunk_size}")
        if self.store_on_gpu:
            logger.info("Storing replay buffer on GPU")

    @classmethod
    def create_from_offline_dataset(cls,
                                   offline_dataset: Dict[str, Any],
                                   capacity: int,
                                   chunk_size: int = 10,
                                   device: Optional[torch.device] = None):
        """
        Create replay buffer initialized with offline dataset.

        Args:
            offline_dataset: Dictionary containing offline trajectories
            capacity: Buffer capacity (should be >= offline dataset size)
            chunk_size: Action chunk size
            device: Device for tensors

        Returns:
            Initialized replay buffer
        """
        buffer = cls(capacity=capacity, chunk_size=chunk_size, device=device)

        # Add offline trajectories
        if 'trajectories' in offline_dataset:
            for trajectory in offline_dataset['trajectories']:
                buffer.add_trajectory(trajectory)
        else:
            # Handle single trajectory format
            buffer.add_trajectory(offline_dataset)

        logger.info(f"Initialized replay buffer with {buffer.size} transitions from offline dataset")
        return buffer

    def add_trajectory(self, trajectory: Dict[str, Any]):
        """
        Add a complete trajectory to the buffer.

        Args:
            trajectory: Dictionary containing:
                - observations: Dict of observation arrays/tensors
                - actions: Action array (T, action_dim)
                - rewards: Reward array (T,)
                - dones: Done flags array (T,)
                - Any additional data
        """
        # Extract trajectory data
        traj_length = len(trajectory['actions'])

        if traj_length == 0:
            return

        # Initialize storage on first trajectory
        if not self.initialized:
            self._initialize_storage(trajectory)

        # Add episode start marker
        if self.size > 0:
            self.episode_starts.append(self.size)

        # Add each transition in the trajectory
        for t in range(traj_length):
            # Extract single step data
            obs = self._extract_step_obs(trajectory, t)
            action = trajectory['actions'][t]
            reward = trajectory['rewards'][t]
            done = trajectory['dones'][t] if t < len(trajectory['dones']) else False

            # Next observation
            if t < traj_length - 1:
                next_obs = self._extract_step_obs(trajectory, t + 1)
            else:
                # Use last observation for terminal state
                next_obs = obs

            # Store in buffer
            self._store_transition(obs, action, reward, done, next_obs)

    def add_transition(self,
                      obs: Dict[str, Any],
                      action: np.ndarray,
                      reward: float,
                      next_obs: Dict[str, Any],
                      done: bool):
        """
        Add a single transition to the buffer in original Q-chunking format.

        Args:
            obs: Current observation (in VLA format with rgb_obs, lang_text, etc.)
            action: Action taken (single action, not chunk)
            reward: Reward received
            next_obs: Next observation (in VLA format)
            done: Whether episode ended
        """
        if not self.initialized:
            # Create dummy trajectory to initialize storage from transition format
            # Extract rgb observation from VLA format
            rgb_obs = obs.get('rgb_obs', {})
            if isinstance(rgb_obs, dict):
                sample_rgb = rgb_obs.get('rgb_static', np.zeros((224, 224, 3)))
            else:
                sample_rgb = rgb_obs

            dummy_traj = {
                'actions': [action],
                'rewards': [reward],
                'dones': [done],
                'rgb_obs': [sample_rgb],
                'instruction': [obs.get('lang_text', 'dummy')]
            }

            # Add gripper cam if available
            if isinstance(rgb_obs, dict) and 'rgb_gripper' in rgb_obs:
                dummy_traj['rgb_gripper'] = [rgb_obs['rgb_gripper']]

            # Add proprioception if available
            if 'robot_obs' in obs:
                dummy_traj['robot_obs'] = [obs['robot_obs']]

            self._initialize_storage(dummy_traj)

        self._store_transition(obs, action, reward, done, next_obs)

    def sample_sequence(self, batch_size: int, discount: float = 0.99) -> Dict[str, torch.Tensor]:
        """
        Sample sequences with proper format matching original VLA dataloader.

        Returns data in the format expected by original VLA interface:
        {
            'rgb_obs': {'rgb_static': ..., 'rgb_gripper': ...},
            'actions': ...,
            'lang_text': ...,
            'robot_obs': ...,
            'rewards': ...,
            'terminals': ...
        }

        Args:
            batch_size: Number of sequences to sample
            discount: Discount factor for reward computation

        Returns:
            Dictionary containing batched sequences in original VLA format
        """
        if self.size < self.chunk_size:
            # Not enough data for sequence sampling
            if self.size == 0:
                raise ValueError("Buffer is empty")
            # Fall back to available data
            available_size = max(1, self.size - 1)
            idxs = np.random.randint(0, available_size, size=batch_size)
            actual_chunk_size = min(self.chunk_size, self.size)
        else:
            # Sample starting indices that allow full sequences
            max_start_idx = self.size - self.chunk_size
            idxs = np.random.randint(0, max_start_idx + 1, size=batch_size)
            actual_chunk_size = self.chunk_size

        # Initialize output in VLA-compatible format
        batch = {
            'rgb_obs': {'rgb_static': []},
            'actions': [],
            'lang_text': [],
            'rewards': [],
            'terminals': []
        }

        # Add optional modalities if available
        if self.has_gripper_cam:
            batch['rgb_obs']['rgb_gripper'] = []
        if self.has_proprio:
            batch['robot_obs'] = []

        # Fill sequences
        for i, start_idx in enumerate(idxs):
            sequence_data = self._extract_sequence(start_idx, actual_chunk_size, discount)

            # Extract observations
            obs = sequence_data['observations']
            next_obs = sequence_data['next_observations']

            # Format RGB observations
            batch['rgb_obs']['rgb_static'].append(torch.from_numpy(obs['rgb_obs'].copy()))
            if self.has_gripper_cam:
                batch['rgb_obs']['rgb_gripper'].append(torch.from_numpy(obs.get('rgb_gripper', obs['rgb_obs']).copy()))

            # Format other data
            batch['actions'].append(torch.from_numpy(sequence_data['actions'].copy()))
            batch['rewards'].append(torch.from_numpy(sequence_data['rewards'].copy()))
            batch['terminals'].append(torch.tensor(sequence_data['terminals'], dtype=torch.bool))
            batch['lang_text'].append(obs['instruction'])

            if self.has_proprio:
                batch['robot_obs'].append(torch.from_numpy(obs.get('proprio', np.zeros(self.proprio_dim)).copy()))

        # Stack tensors
        batch['rgb_obs']['rgb_static'] = torch.stack(batch['rgb_obs']['rgb_static']).to(self.device)
        if self.has_gripper_cam:
            batch['rgb_obs']['rgb_gripper'] = torch.stack(batch['rgb_obs']['rgb_gripper']).to(self.device)

        batch['actions'] = torch.stack(batch['actions']).to(self.device)
        batch['rewards'] = torch.stack(batch['rewards']).to(self.device)
        batch['terminals'] = torch.stack(batch['terminals']).to(self.device)

        if self.has_proprio:
            batch['robot_obs'] = torch.stack(batch['robot_obs']).to(self.device)

        return batch

    def _extract_sequence(self, start_idx: int, sequence_length: int, discount: float) -> Dict[str, Any]:
        """
        Extract a sequence starting at start_idx with proper discount computation.

        This follows the original Q-chunking approach:
        - Computes cumulative discounted rewards
        - Tracks validity across episode boundaries
        - Handles terminal states correctly
        """
        sequence = {}

        # Extract observations (first step of sequence)
        sequence['observations'] = self._get_observation_at_index(start_idx)

        # Initialize arrays for sequence data
        actions = np.zeros((sequence_length, self.action_dim))
        rewards = np.zeros((sequence_length,))
        terminals = np.zeros((sequence_length,), dtype=bool)
        valid = np.ones((sequence_length,), dtype=bool)

        # Fill sequence data with discount computation
        cumulative_reward = 0.0
        for i in range(sequence_length):
            idx = start_idx + i

            if idx >= self.size:
                # Beyond buffer size - mark as invalid
                valid[i] = False
                actions[i] = actions[i-1] if i > 0 else 0
                rewards[i] = cumulative_reward
                terminals[i] = True
                continue

            # Get data for this step
            actions[i] = self.actions[idx]
            step_reward = self.rewards[idx]
            terminals[i] = self.terminals[idx]

            # Update cumulative reward
            if i == 0:
                cumulative_reward = step_reward
            else:
                # Check if we crossed episode boundary
                if self._crossed_episode_boundary(start_idx, i):
                    # Reset cumulative reward for new episode
                    cumulative_reward = step_reward
                    valid[i] = False  # Mark transition across episodes as invalid
                else:
                    cumulative_reward += step_reward * (discount ** i)

            rewards[i] = cumulative_reward

            # If terminal, break sequence
            if terminals[i]:
                for j in range(i + 1, sequence_length):
                    valid[j] = False
                    actions[j] = actions[i]  # Repeat last action
                    rewards[j] = cumulative_reward
                    terminals[j] = True
                break

        sequence['actions'] = actions
        sequence['rewards'] = rewards
        sequence['terminals'] = terminals[-1]  # Terminal state of sequence
        sequence['valid'] = valid
        sequence['next_observations'] = self._get_observation_at_index(
            min(start_idx + sequence_length, self.size - 1)
        )

        return sequence

    def _initialize_storage(self, sample_trajectory: Dict[str, Any]):
        """Initialize storage arrays based on first trajectory."""
        traj_length = len(sample_trajectory['actions'])

        # Determine dimensions from sample data
        sample_action = sample_trajectory['actions'][0]
        self.action_dim = sample_action.shape[-1] if hasattr(sample_action, 'shape') else len(sample_action)

        # Handle RGB observation shape
        sample_rgb = sample_trajectory.get('rgb_obs', [np.zeros((224, 224, 3))])[0]
        if hasattr(sample_rgb, 'shape'):
            self.rgb_shape = sample_rgb.shape
        else:
            self.rgb_shape = (224, 224, 3)  # Default LIBERO image shape

        # Check for proprioception (robot_obs or proprio)
        self.has_proprio = 'robot_obs' in sample_trajectory or 'proprio' in sample_trajectory
        if self.has_proprio:
            if 'robot_obs' in sample_trajectory:
                sample_proprio = sample_trajectory['robot_obs'][0]
            else:
                sample_proprio = sample_trajectory['proprio'][0]
            self.proprio_dim = sample_proprio.shape[-1] if hasattr(sample_proprio, 'shape') else len(sample_proprio)

        # Check for second RGB camera
        self.has_gripper_cam = 'rgb_gripper' in sample_trajectory

        # Initialize storage arrays
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.terminals = np.zeros((self.capacity,), dtype=bool)

        # Observations storage - primary RGB camera
        self.rgb_obs = np.zeros((self.capacity,) + self.rgb_shape, dtype=np.float32)
        self.next_rgb_obs = np.zeros((self.capacity,) + self.rgb_shape, dtype=np.float32)

        # Secondary RGB camera (gripper cam) if available
        if self.has_gripper_cam:
            self.rgb_gripper = np.zeros((self.capacity,) + self.rgb_shape, dtype=np.float32)
            self.next_rgb_gripper = np.zeros((self.capacity,) + self.rgb_shape, dtype=np.float32)

        # Proprioception storage
        if self.has_proprio:
            self.proprio = np.zeros((self.capacity, self.proprio_dim), dtype=np.float32)
            self.next_proprio = np.zeros((self.capacity, self.proprio_dim), dtype=np.float32)

        # Instructions (stored as list since they're strings)
        self.instructions = [''] * self.capacity
        self.next_instructions = [''] * self.capacity

        self.initialized = True
        logger.info(f"Initialized storage: action_dim={self.action_dim}, rgb_shape={self.rgb_shape}, "
                   f"has_proprio={self.has_proprio}, capacity={self.capacity}")

    def _extract_step_obs(self, trajectory: Dict[str, Any], step: int) -> Dict[str, Any]:
        """Extract observation at specific step from trajectory."""
        obs = {}

        # Handle RGB observations (primary camera)
        if 'rgb_obs' in trajectory:
            obs['rgb_obs'] = trajectory['rgb_obs'][step]

        # Handle second RGB camera if available
        if 'rgb_gripper' in trajectory:
            obs['rgb_gripper'] = trajectory['rgb_gripper'][step]

        # Handle proprioception/robot state
        if 'robot_obs' in trajectory:
            obs['proprio'] = trajectory['robot_obs'][step]
        elif 'proprio' in trajectory:
            obs['proprio'] = trajectory['proprio'][step]

        # Handle language instruction
        if 'instruction' in trajectory:
            if isinstance(trajectory['instruction'], list):
                obs['instruction'] = trajectory['instruction'][step] if step < len(trajectory['instruction']) else trajectory['instruction'][-1]
            else:
                obs['instruction'] = trajectory['instruction']
        else:
            obs['instruction'] = 'manipulate object'

        return obs

    def _store_transition(self, obs: Dict[str, Any], action: np.ndarray, reward: float, done: bool, next_obs: Dict[str, Any]):
        """Store a single transition in the buffer."""
        idx = self.ptr

        # Store action, reward, terminal
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.terminals[idx] = done

        # Handle VLA observation format: obs['rgb_obs']['rgb_static']
        current_rgb_obs = obs.get('rgb_obs', {})
        next_rgb_obs = next_obs.get('rgb_obs', {})

        # Store RGB observations (primary camera)
        if isinstance(current_rgb_obs, dict):
            self.rgb_obs[idx] = current_rgb_obs.get('rgb_static', np.zeros(self.rgb_shape))
        else:
            self.rgb_obs[idx] = current_rgb_obs

        if isinstance(next_rgb_obs, dict):
            self.next_rgb_obs[idx] = next_rgb_obs.get('rgb_static', np.zeros(self.rgb_shape))
        else:
            self.next_rgb_obs[idx] = next_rgb_obs

        # Store RGB gripper camera if available
        if self.has_gripper_cam:
            if isinstance(current_rgb_obs, dict) and 'rgb_gripper' in current_rgb_obs:
                self.rgb_gripper[idx] = current_rgb_obs['rgb_gripper']
            else:
                self.rgb_gripper[idx] = np.zeros(self.rgb_shape)

            if isinstance(next_rgb_obs, dict) and 'rgb_gripper' in next_rgb_obs:
                self.next_rgb_gripper[idx] = next_rgb_obs['rgb_gripper']
            else:
                self.next_rgb_gripper[idx] = np.zeros(self.rgb_shape)

        # Store proprioception
        if self.has_proprio:
            if 'robot_obs' in obs:
                self.proprio[idx] = obs['robot_obs']
            elif 'proprio' in obs:
                self.proprio[idx] = obs['proprio']
            else:
                self.proprio[idx] = np.zeros(self.proprio_dim)

            if 'robot_obs' in next_obs:
                self.next_proprio[idx] = next_obs['robot_obs']
            elif 'proprio' in next_obs:
                self.next_proprio[idx] = next_obs['proprio']
            else:
                self.next_proprio[idx] = np.zeros(self.proprio_dim)

        # Store language instructions
        self.instructions[idx] = obs.get('lang_text', obs.get('instruction', 'manipulate object'))
        self.next_instructions[idx] = next_obs.get('lang_text', next_obs.get('instruction', 'manipulate object'))

        # Update pointers
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _get_observation_at_index(self, idx: int) -> Dict[str, Any]:
        """Get observation at specific buffer index."""
        obs = {
            'rgb_obs': self.rgb_obs[idx],
            'instruction': self.instructions[idx]
        }

        # Add gripper camera if available
        if self.has_gripper_cam:
            obs['rgb_gripper'] = self.rgb_gripper[idx]

        # Add proprioception if available
        if self.has_proprio:
            obs['proprio'] = self.proprio[idx]

        return obs

    def _crossed_episode_boundary(self, start_idx: int, offset: int) -> bool:
        """Check if we crossed an episode boundary within a sequence."""
        current_idx = start_idx + offset

        # Check if any episode start marker is between start_idx and current_idx
        for episode_start in self.episode_starts:
            if start_idx < episode_start <= current_idx:
                return True
        return False

    def _initialize_batch_tensors(self, batch_size: int, sequence_length: int) -> Dict[str, torch.Tensor]:
        """Initialize batch tensors for sequence sampling."""
        batch = {}

        # RGB observations
        batch['rgb_obs'] = torch.zeros((batch_size,) + self.rgb_shape, dtype=torch.float32, device=self.device)
        batch['next_rgb_obs'] = torch.zeros((batch_size,) + self.rgb_shape, dtype=torch.float32, device=self.device)

        # Actions and rewards
        batch['actions'] = torch.zeros((batch_size, sequence_length, self.action_dim), dtype=torch.float32, device=self.device)
        batch['rewards'] = torch.zeros((batch_size, sequence_length), dtype=torch.float32, device=self.device)
        batch['terminals'] = torch.zeros((batch_size,), dtype=torch.bool, device=self.device)
        batch['valid'] = torch.ones((batch_size, sequence_length), dtype=torch.bool, device=self.device)

        # Proprioception
        if self.has_proprio:
            batch['proprio'] = torch.zeros((batch_size, self.proprio_dim), dtype=torch.float32, device=self.device)
            batch['next_proprio'] = torch.zeros((batch_size, self.proprio_dim), dtype=torch.float32, device=self.device)

        # Instructions
        batch['instruction'] = []
        batch['next_instruction'] = []

        return batch

    def _fill_batch_entry(self, batch: Dict[str, torch.Tensor], batch_idx: int, sequence_data: Dict[str, Any]):
        """Fill a single entry in the batch with sequence data."""
        obs = sequence_data['observations']
        next_obs = sequence_data['next_observations']

        # Fill observations
        if isinstance(obs['rgb_obs'], np.ndarray):
            batch['rgb_obs'][batch_idx] = torch.from_numpy(obs['rgb_obs'].copy())
        else:
            batch['rgb_obs'][batch_idx] = obs['rgb_obs']

        if isinstance(next_obs['rgb_obs'], np.ndarray):
            batch['next_rgb_obs'][batch_idx] = torch.from_numpy(next_obs['rgb_obs'].copy())
        else:
            batch['next_rgb_obs'][batch_idx] = next_obs['rgb_obs']

        # Fill proprioception if available
        if self.has_proprio and 'proprio' in obs:
            if isinstance(obs['proprio'], np.ndarray):
                batch['proprio'][batch_idx] = torch.from_numpy(obs['proprio'].copy())
            else:
                batch['proprio'][batch_idx] = obs['proprio']

        if self.has_proprio and 'proprio' in next_obs:
            if isinstance(next_obs['proprio'], np.ndarray):
                batch['next_proprio'][batch_idx] = torch.from_numpy(next_obs['proprio'].copy())
            else:
                batch['next_proprio'][batch_idx] = next_obs['proprio']

        # Fill sequence data
        batch['actions'][batch_idx] = torch.from_numpy(sequence_data['actions'].copy())
        batch['rewards'][batch_idx] = torch.from_numpy(sequence_data['rewards'].copy())
        # Handle terminals - can be a single bool or an array
        terminals = sequence_data['terminals']
        if isinstance(terminals, np.ndarray):
            batch['terminals'][batch_idx] = torch.from_numpy(terminals.copy())
        else:
            # Single boolean value - convert to tensor
            batch['terminals'][batch_idx] = torch.tensor(terminals, dtype=torch.bool)
        batch['valid'][batch_idx] = torch.from_numpy(sequence_data['valid'].copy())

        # Fill instructions
        batch['instruction'].append(obs['instruction'])
        batch['next_instruction'].append(next_obs['instruction'])

    def sample_recent(self, batch_size: int, recent_fraction: float = 0.5, discount: float = 0.99) -> Dict[str, torch.Tensor]:
        """
        Sample with bias towards recent transitions.

        Args:
            batch_size: Number of transitions to sample
            recent_fraction: Fraction of batch from recent transitions
            discount: Discount factor for reward computation

        Returns:
            Batch of sequences biased towards recent data
        """
        if self.size < self.chunk_size:
            return self.sample_sequence(batch_size, discount)

        # Split batch between recent and old data
        recent_count = int(batch_size * recent_fraction)
        old_count = batch_size - recent_count

        # Sample from recent portion (last quarter of buffer)
        recent_start = max(0, self.size - self.size // 4)
        recent_end = max(recent_start + 1, self.size - self.chunk_size)

        if recent_end > recent_start:
            recent_idxs = np.random.randint(recent_start, recent_end, size=recent_count)
        else:
            recent_idxs = np.array([], dtype=int)

        # Sample from older data
        old_end = max(1, self.size - self.chunk_size - self.size // 4)
        if old_end > 0:
            old_idxs = np.random.randint(0, old_end, size=old_count)
        else:
            old_idxs = np.array([], dtype=int)

        # Combine indices
        all_idxs = np.concatenate([recent_idxs, old_idxs]) if len(old_idxs) > 0 else recent_idxs

        if len(all_idxs) < batch_size:
            # Fall back to regular sampling
            return self.sample_sequence(batch_size, discount)

        # Create batch
        batch = self._initialize_batch_tensors(batch_size, self.chunk_size)

        for i, start_idx in enumerate(all_idxs):
            sequence_data = self._extract_sequence(start_idx, self.chunk_size, discount)
            self._fill_batch_entry(batch, i, sequence_data)

        return batch

    def __len__(self):
        return self.size

    def is_empty(self):
        return self.size == 0

    def get_state(self) -> Dict[str, Any]:
        """Get replay buffer state for checkpointing."""
        if not self.initialized:
            return None

        state = {
            'size': self.size,
            'ptr': self.ptr,
            'initialized': self.initialized,
            'capacity': self.capacity,
            'chunk_size': self.chunk_size,
            'action_dim': self.action_dim,
            'rgb_shape': self.rgb_shape,
            'has_proprio': self.has_proprio,
            'has_gripper_cam': self.has_gripper_cam,
            'proprio_dim': self.proprio_dim if self.has_proprio else None,
            'episode_starts': self.episode_starts.copy(),
        }

        # Only save data if we have some
        if self.size > 0:
            state.update({
                'actions': self.actions[:self.size].copy(),
                'rewards': self.rewards[:self.size].copy(),
                'terminals': self.terminals[:self.size].copy(),
                'rgb_obs': self.rgb_obs[:self.size].copy(),
                'next_rgb_obs': self.next_rgb_obs[:self.size].copy(),
                'instructions': self.instructions[:self.size].copy(),
                'next_instructions': self.next_instructions[:self.size].copy(),
            })

            if self.has_proprio:
                state.update({
                    'proprio': self.proprio[:self.size].copy(),
                    'next_proprio': self.next_proprio[:self.size].copy(),
                })

            if self.has_gripper_cam:
                state.update({
                    'rgb_gripper': self.rgb_gripper[:self.size].copy(),
                    'next_rgb_gripper': self.next_rgb_gripper[:self.size].copy(),
                })

        return state

    def load_state(self, state: Dict[str, Any]):
        """Load replay buffer state from checkpoint."""
        if state is None:
            return

        # Restore basic parameters
        self.size = state['size']
        self.ptr = state['ptr']
        self.initialized = state['initialized']
        self.capacity = state.get('capacity', self.capacity)
        self.chunk_size = state.get('chunk_size', self.chunk_size)
        self.action_dim = state['action_dim']
        self.rgb_shape = tuple(state['rgb_shape'])
        self.has_proprio = state['has_proprio']
        self.has_gripper_cam = state.get('has_gripper_cam', False)
        self.proprio_dim = state.get('proprio_dim')
        self.episode_starts = state.get('episode_starts', [])

        if not self.initialized:
            return

        # Initialize storage arrays
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.terminals = np.zeros((self.capacity,), dtype=bool)
        self.rgb_obs = np.zeros((self.capacity,) + self.rgb_shape, dtype=np.float32)
        self.next_rgb_obs = np.zeros((self.capacity,) + self.rgb_shape, dtype=np.float32)
        self.instructions = [''] * self.capacity
        self.next_instructions = [''] * self.capacity

        if self.has_gripper_cam:
            self.rgb_gripper = np.zeros((self.capacity,) + self.rgb_shape, dtype=np.float32)
            self.next_rgb_gripper = np.zeros((self.capacity,) + self.rgb_shape, dtype=np.float32)

        if self.has_proprio:
            self.proprio = np.zeros((self.capacity, self.proprio_dim), dtype=np.float32)
            self.next_proprio = np.zeros((self.capacity, self.proprio_dim), dtype=np.float32)

        # Restore data if available
        if self.size > 0:
            if 'actions' in state:
                self.actions[:self.size] = state['actions']
            if 'rewards' in state:
                self.rewards[:self.size] = state['rewards']
            if 'terminals' in state:
                self.terminals[:self.size] = state['terminals']
            if 'rgb_obs' in state:
                self.rgb_obs[:self.size] = state['rgb_obs']
            if 'next_rgb_obs' in state:
                self.next_rgb_obs[:self.size] = state['next_rgb_obs']
            if 'instructions' in state:
                self.instructions[:self.size] = state['instructions']
            if 'next_instructions' in state:
                self.next_instructions[:self.size] = state['next_instructions']

            if self.has_proprio:
                if 'proprio' in state:
                    self.proprio[:self.size] = state['proprio']
                if 'next_proprio' in state:
                    self.next_proprio[:self.size] = state['next_proprio']

            if self.has_gripper_cam:
                if 'rgb_gripper' in state:
                    self.rgb_gripper[:self.size] = state['rgb_gripper']
                if 'next_rgb_gripper' in state:
                    self.next_rgb_gripper[:self.size] = state['next_rgb_gripper']

        logger.info(f"Loaded replay buffer state: size={self.size}, capacity={self.capacity}")


# Backward compatibility alias
ReplayBuffer = SequenceReplayBuffer