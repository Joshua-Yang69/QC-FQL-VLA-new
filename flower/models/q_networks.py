import torch
import torch.nn as nn
from typing import Tuple, Optional
import logging



logger = logging.getLogger(__name__)



class QNetwork(nn.Module):
    """
    Single Q-network for value estimation.

    Takes state features and action chunks as input,
    outputs Q-value for the state-action pair.
    """

    def __init__(self,
                 state_dim: int,
                 total_action_dim: int,
                 hidden_dim: int = 512,
                 n_layers: int = 3,
                 dropout: float = 0.1):
        """
        Initialize Q-network.

        Args:
            state_dim: Dimension of state features
            action_dim: Dimension of actions (total for chunk)
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
            dropout: Dropout rate
        """
        super().__init__()

        self.state_dim = state_dim
        self.total_action_dim = total_action_dim


        # Build network
        layers = []
        input_dim = state_dim + total_action_dim


        for i in range(n_layers):
            layers.extend([
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        #output:(B,1)

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)


    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):

            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for state-action pairs.

        Args:
            states: State features (B, state_dim)
            actions: Action chunks (B, chunk_size, action_dim) or (B, chunk_size * action_dim)

        Returns:
            Q-values (B, 1)
        """
        # Flatten action chunks if needed
        if actions.dim() == 3:  # (B, chunk_size, action_dim)
            B, C, A = actions.shape
            actions = actions.reshape(B, -1)  # (B, chunk_size * action_dim)

        # Concatenate state and actions
        x = torch.cat([states, actions], dim=-1)

        # Forward through network
        q_values = self.network(x)

        return q_values


class DoubleQNetwork(nn.Module):
    """
    Double Q-network for reduced overestimation bias.

    Maintains two Q-networks and uses the minimum for conservative estimates.
    This is crucial for stable Q-learning with function approximation.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 chunk_size: int = 10,  #current_setting
                 hidden_dim: int = 512,
                 n_layers: int = 3,
                 dropout: float = 0.1):
        """
        Initialize double Q-network.

        Args:
            state_dim: Dimension of state features
            action_dim: Dimension of single action
            chunk_size: Size of action chunks
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
            dropout: Dropout rate
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

        # Total action dimension for chunks
        total_action_dim = action_dim * chunk_size

        # Two Q-networks
        self.q1 = QNetwork(
            state_dim=state_dim,
            total_action_dim=total_action_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        )

        self.q2 = QNetwork(
            state_dim=state_dim,
            total_action_dim=total_action_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        )


    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q-values from both networks.

        Args:
            states: State features (B, state_dim)
            actions: Action chunks (B, chunk_size, action_dim)

        Returns:
            Tuple of (q1_values, q2_values), each (B, 1)
        """
        q1_values = self.q1(states, actions)
        q2_values = self.q2(states, actions)

        return q1_values, q2_values

    def q_min(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Get minimum Q-value for conservative estimation.

        Args:
            states: State features (B, state_dim)
            actions: Action chunks (B, chunk_size, action_dim)

        Returns:
            Minimum Q-values (B, 1)
        """
        q1, q2 = self.forward(states, actions)
        return torch.min(q1, q2)

    def q_mean(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Get mean Q-value.

        Args:
            states: State features (B, state_dim)
            actions: Action chunks (B, chunk_size, action_dim)

        Returns:
            Mean Q-values (B, 1)
        """
        q1, q2 = self.forward(states, actions)
        return (q1 + q2) / 2


class EnsembleQNetwork(nn.Module):
    """
    Ensemble of Q-networks for uncertainty estimation.

    Can be used for exploration or uncertainty-aware planning.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 chunk_size: int = 10,
                 ensemble_size: int = 5,
                 hidden_dim: int = 512,
                 n_layers: int = 3,
                 dropout: float = 0.1):
        """
        Initialize ensemble Q-network.

        Args:
            state_dim: Dimension of state features
            action_dim: Dimension of single action
            chunk_size: Size of action chunks
            ensemble_size: Number of Q-networks in ensemble
            hidden_dim: Hidden layer dimension
            n_layers: Number of hidden layers
            dropout: Dropout rate
        """
        super().__init__()

        self.ensemble_size = ensemble_size
        self.chunk_size = chunk_size

        # Create ensemble
        total_action_dim = action_dim * chunk_size

        self.q_networks = nn.ModuleList([
            QNetwork(
                state_dim=state_dim,
                total_action_dim=total_action_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                dropout=dropout
            )
            for _ in range(ensemble_size)
        ])

        logger.info(f"Created Ensemble Q-Network with {ensemble_size} members")

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values from all networks.

        Args:
            states: State features (B, state_dim)
            actions: Action chunks (B, chunk_size, action_dim)

        Returns:
            Q-values from all networks (B, ensemble_size)
        """
        q_values = []

        for q_net in self.q_networks:
            q = q_net(states, actions)
            q_values.append(q)

        # Stack along ensemble dimension
        q_values = torch.cat(q_values, dim=-1)  # (B, ensemble_size)

        return q_values

    def q_mean(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get mean Q-value across ensemble."""
        q_values = self.forward(states, actions)
        return q_values.mean(dim=-1, keepdim=True)

    def q_std(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get standard deviation of Q-values (uncertainty)."""
        q_values = self.forward(states, actions)
        return q_values.std(dim=-1, keepdim=True)

    def q_ucb(self, states: torch.Tensor, actions: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
        """Get upper confidence bound Q-value for exploration."""
        q_mean = self.q_mean(states, actions)
        q_std = self.q_std(states, actions)
        return q_mean + beta * q_std


# Export
__all__ = ['QNetwork', 'DoubleQNetwork', 'EnsembleQNetwork']