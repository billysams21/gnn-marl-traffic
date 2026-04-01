"""
Q-Network (Double DQN) for action-value estimation.

Takes spatially-enriched embeddings from GAT and outputs Q-values for each action.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Q-Network MLP.

    Architecture (from ARSITEKTUR_GNN_MARL.md):
        Input: embed_dim (h'_i from GAT)
        Hidden: 128 -> 64
        Output: num_actions (Q-value per action)

    Shared weights across all agents.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_actions: int = 4,
        hidden_dims: list = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        in_dim = embed_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Embedding from GAT encoder [batch_nodes, embed_dim]
               or [num_nodes, embed_dim] for single-step

        Returns:
            q_values: [batch_nodes, num_actions]
        """
        return self.network(h)


class PredictionHead(nn.Module):
    """
    Auxiliary Prediction Head for next-state prediction.

    Architecture (from ARSITEKTUR_GNN_MARL.md):
        Input: embed_dim (h'_i) + action_embed_dim = 72
        Hidden: 128 -> 64
        Output: 2 (simplified) or obs_dim (full) predicted values

    Two prediction modes:
    - 'simplified' (proposal §4.5): Predict [avg_queue, avg_density] per agent
      - More stable, focuses on key metrics
      - Output: 2 values per agent
    - 'full' (original): Predict entire next state vector
      - More learning signal, but potentially noisy
      - Output: obs_dim values per agent

    Provides auxiliary learning signal and interpretability.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_actions: int = 4,
        action_embed_dim: int = 8,
        obs_dim: int = 25,
        hidden_dims: list = None,
        prediction_mode: str = "simplified",  # 'simplified' or 'full'
        num_lanes: int = 8,  # for computing average metrics
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.obs_dim = obs_dim
        self.prediction_mode = prediction_mode
        self.num_lanes = num_lanes

        # Output dimension depends on mode
        if prediction_mode == "simplified":
            # Predict: [avg_queue, avg_density] = 2 values
            output_dim = 2
        else:
            # Predict: full observation
            output_dim = obs_dim

        self.action_embedding = nn.Embedding(num_actions, action_embed_dim)

        layers = []
        in_dim = embed_dim + action_embed_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Embedding from GAT [batch_nodes, embed_dim]
            actions: Action indices [batch_nodes] (long)

        Returns:
            If 'simplified': [batch_nodes, 2] (avg_queue, avg_density)
            If 'full': [batch_nodes, obs_dim] (predicted next state)
        """
        action_emb = self.action_embedding(actions)
        x = torch.cat([h, action_emb], dim=-1)
        return self.network(x)

    def compute_target(self, next_obs: torch.Tensor, num_lanes: int = None) -> torch.Tensor:
        """
        Compute prediction target from observation.
        
        Args:
            next_obs: Next observation [batch_nodes, obs_dim]
            num_lanes: Number of lanes per intersection (for averaging)
            
        Returns:
            If 'simplified': [batch_nodes, 2] (avg_queue_norm, avg_density_norm)
            If 'full': [batch_nodes, obs_dim] (unchanged)
        """
        if self.prediction_mode != "simplified":
            return next_obs
        
        # For simplified mode: extract and average queue and density
        # Observation format: [queue, delta_queue, density, waiting, delta_density, phase, duration]
        # queue is first num_lanes elements, density is next num_lanes
        lanes = num_lanes if num_lanes is not None else self.num_lanes
        
        if next_obs.shape[-1] < 2 * lanes:
            # Not enough dimensions, fall back to first 2 values
            return next_obs[:, :2]
        
        # Average queue per agent (normalized, already in [0,1])
        avg_queue = next_obs[:, :lanes].mean(dim=-1, keepdim=True)
        # Average density per agent (already in [0,1])
        avg_density = next_obs[:, lanes:2*lanes].mean(dim=-1, keepdim=True)
        
        return torch.cat([avg_queue, avg_density], dim=-1)
