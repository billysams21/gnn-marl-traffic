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
        Output: obs_dim (predicted next state)

    Provides auxiliary learning signal and interpretability.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_actions: int = 4,
        action_embed_dim: int = 8,
        obs_dim: int = 25,
        hidden_dims: list = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.action_embedding = nn.Embedding(num_actions, action_embed_dim)

        layers = []
        in_dim = embed_dim + action_embed_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, obs_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Embedding from GAT [batch_nodes, embed_dim]
            actions: Action indices [batch_nodes] (long)

        Returns:
            predicted_next_state: [batch_nodes, obs_dim]
        """
        action_emb = self.action_embedding(actions)
        x = torch.cat([h, action_emb], dim=-1)
        return self.network(x)
