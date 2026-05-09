"""
GAT-Double DQN Agent for multi-agent traffic signal control.

Combines:
- GAT Encoder: spatial information aggregation across intersections
- Double DQN: action-value estimation with reduced overestimation
- Prediction Head: auxiliary next-state prediction task
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, Optional
import copy

from src.models.gat_encoder import GATEncoder
from src.models.q_network import QNetwork, PredictionHead
from src.agents.replay_buffer import ReplayBuffer


def _move_optimizer_state_to_device(optimizer: optim.Optimizer, device: torch.device):
    """Move optimizer state tensors to the active device after checkpoint load."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


class GATDoubleDQNAgent:
    """
    GAT-Double DQN agent with shared weights across all traffic lights.

    Training objective: L_total = L_RL + lambda * L_prediction
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        num_agents: int,
        edge_index: np.ndarray,
        # GAT params
        gat_hidden_dim: int = 64,
        gat_embed_dim: int = 64,
        gat_num_heads: int = 4,
        gat_dropout: float = 0.0,
        # Q-Network params
        q_hidden_dims: list = None,
        # Prediction Head params
        pred_lambda: float = 0.3,
        action_embed_dim: int = 8,
        prediction_mode: str = "full",
        # RL params
        lr: float = 3e-4,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.997,
        batch_size: int = 64,
        buffer_size: int = 50000,
        target_update_freq: int = 1000,
        # Device
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.num_agents = num_agents
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.pred_lambda = pred_lambda

        # Edge index for graph (constant topology)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device)

        # ---- Networks ----
        # GAT Encoder (shared)
        self.gat_encoder = GATEncoder(
            obs_dim=obs_dim,
            hidden_dim=gat_hidden_dim,
            embed_dim=gat_embed_dim,
            num_heads=gat_num_heads,
            dropout=gat_dropout,
        ).to(self.device)

        # Q-Network (shared, online)
        self.q_network = QNetwork(
            embed_dim=gat_embed_dim,
            num_actions=num_actions,
            hidden_dims=q_hidden_dims,
        ).to(self.device)

        # Target networks (for Double DQN)
        self.target_gat_encoder = copy.deepcopy(self.gat_encoder).to(self.device)
        self.target_q_network = copy.deepcopy(self.q_network).to(self.device)
        self.target_gat_encoder.eval()
        self.target_q_network.eval()

        # Prediction Head (auxiliary task)
        # prediction_mode controls output: 'simplified' = [avg_queue, avg_density], 'full' = obs_dim
        self.prediction_head = PredictionHead(
            embed_dim=gat_embed_dim,
            num_actions=num_actions,
            action_embed_dim=action_embed_dim,
            obs_dim=obs_dim,
            prediction_mode=prediction_mode,
            num_lanes=8,  # Will be updated from env if needed
        ).to(self.device)

        # ---- Optimizer ----
        self.optimizer = optim.Adam(
            list(self.gat_encoder.parameters())
            + list(self.q_network.parameters())
            + list(self.prediction_head.parameters()),
            lr=lr,
        )

        # ---- Replay Buffer ----
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # Step counter for target updates
        self._train_steps = 0

    def set_num_lanes(self, num_lanes: int):
        """Update number of lanes for prediction head (for simplified mode)."""
        self.prediction_head.num_lanes = num_lanes

    def select_actions(
        self, observations: np.ndarray, evaluate: bool = False
    ) -> np.ndarray:
        """
        Select actions for all agents using epsilon-greedy.

        Args:
            observations: [num_agents, obs_dim]
            evaluate: If True, use greedy policy (no exploration)

        Returns:
            actions: [num_agents] action indices
        """
        if not evaluate and np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions, size=self.num_agents)

        with torch.no_grad():
            obs_tensor = torch.tensor(observations, dtype=torch.float32).to(self.device)
            embeddings = self.gat_encoder(obs_tensor, self.edge_index)
            q_values = self.q_network(embeddings)
            actions = q_values.argmax(dim=-1).cpu().numpy()

        return actions

    def store_transition(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        done: bool,
    ):
        """Store a graph-level transition in replay buffer."""
        self.replay_buffer.push(states, actions, rewards, next_states, done)

    def _build_batched_edge_index(self, batch_size: int) -> torch.Tensor:
        """Build batched edge_index for PyG-style batching (block-diagonal graphs)."""
        edge_indices = []
        num_nodes = self.num_agents
        for b in range(batch_size):
            edge_indices.append(self.edge_index + b * num_nodes)
        return torch.cat(edge_indices, dim=1)

    def train_step(self) -> Optional[Dict[str, float]]:
        """
        Perform one training step (sample batch, compute loss, update).
        Uses batched graph computation for efficiency.

        Returns:
            Dict of loss values, or None if buffer too small.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(self.device)

        B, N = states_t.shape[0], states_t.shape[1]  # batch_size, num_agents

        # Flatten: [B, N, obs_dim] -> [B*N, obs_dim]
        states_flat = states_t.reshape(B * N, -1)
        next_states_flat = next_states_t.reshape(B * N, -1)
        actions_flat = actions_t.reshape(B * N)
        rewards_flat = rewards_t.reshape(B * N)
        dones_flat = dones_t.unsqueeze(1).expand(-1, N).reshape(B * N)

        # Build batched edge index (block-diagonal)
        batched_edge_index = self._build_batched_edge_index(B)

        # ---- Forward pass (all graphs at once) ----
        embeddings = self.gat_encoder(states_flat, batched_edge_index)
        q_values = self.q_network(embeddings)  # [B*N, num_actions]
        q_taken = q_values.gather(1, actions_flat.unsqueeze(1)).squeeze(1)  # [B*N]

        # ---- Double DQN Target ----
        with torch.no_grad():
            next_emb_online = self.gat_encoder(next_states_flat, batched_edge_index)
            next_q_online = self.q_network(next_emb_online)
            best_actions = next_q_online.argmax(dim=-1)

            next_emb_target = self.target_gat_encoder(next_states_flat, batched_edge_index)
            next_q_target = self.target_q_network(next_emb_target)
            next_q_best = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

            target = rewards_flat + self.gamma * next_q_best * (1 - dones_flat)

        rl_loss = nn.functional.mse_loss(q_taken, target)

        # ---- Prediction Head Loss ----
        # Target format is determined by prediction_mode:
        # - full: use full next observation
        # - simplified: use aggregated [avg_queue, avg_density]
        predicted_next = self.prediction_head(embeddings, actions_flat)
        pred_target = self.prediction_head.compute_target(next_states_flat)
        pred_loss = nn.functional.mse_loss(predicted_next, pred_target)

        # Combined loss: L_total = L_RL + lambda * L_pred
        total_loss = rl_loss + self.pred_lambda * pred_loss

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            list(self.gat_encoder.parameters())
            + list(self.q_network.parameters())
            + list(self.prediction_head.parameters()),
            max_norm=10.0,
        )
        self.optimizer.step()

        # Update target networks
        self._train_steps += 1
        if self._train_steps % self.target_update_freq == 0:
            self.target_gat_encoder.load_state_dict(self.gat_encoder.state_dict())
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        return {
            "loss_total": total_loss.item(),
            "loss_rl": rl_loss.item(),
            "loss_pred": pred_loss.item(),
        }

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_attention_weights(self, observations: np.ndarray):
        """Get GAT attention weights for interpretability analysis."""
        with torch.no_grad():
            obs_tensor = torch.tensor(observations, dtype=torch.float32).to(self.device)
            _, attn_weights = self.gat_encoder(
                obs_tensor, self.edge_index, return_attention=True
            )
        return attn_weights.cpu().numpy()

    def save(self, path: str, extra_state: Optional[Dict[str, Any]] = None):
        """Save model checkpoint."""
        checkpoint = {
            "gat_encoder": self.gat_encoder.state_dict(),
            "q_network": self.q_network.state_dict(),
            "prediction_head": self.prediction_head.state_dict(),
            "target_gat_encoder": self.target_gat_encoder.state_dict(),
            "target_q_network": self.target_q_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_steps": self._train_steps,
            "replay_buffer": self.replay_buffer.state_dict(),
        }
        if extra_state:
            checkpoint.update(extra_state)
        torch.save(checkpoint, path)

    def load(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        # Always load checkpoints on CPU first so non-model tensors (e.g., RNG state)
        # remain compatible with torch.set_rng_state during resume.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.gat_encoder.load_state_dict(checkpoint["gat_encoder"])
        self.q_network.load_state_dict(checkpoint["q_network"])

        # Backward compatibility for checkpoints across prediction head shape changes.
        # For evaluation, the policy uses GAT + Q-network only, so a partial/skip load is safe.
        if "prediction_head" in checkpoint:
            saved_pred = checkpoint["prediction_head"]
            current_pred = self.prediction_head.state_dict()
            compatible = {
                k: v
                for k, v in saved_pred.items()
                if k in current_pred and current_pred[k].shape == v.shape
            }
            current_pred.update(compatible)
            self.prediction_head.load_state_dict(current_pred)

        self.target_gat_encoder.load_state_dict(checkpoint["target_gat_encoder"])
        self.target_q_network.load_state_dict(checkpoint["target_q_network"])
        if "optimizer" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                _move_optimizer_state_to_device(self.optimizer, self.device)
            except ValueError:
                # Ignore optimizer incompatibility across architecture changes.
                pass
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self._train_steps = checkpoint.get("train_steps", self._train_steps)
        if "replay_buffer" in checkpoint:
            self.replay_buffer.load_state_dict(checkpoint["replay_buffer"])

        return checkpoint


class IndependentDQNAgent:
    """
    Independent DQN baseline agent (no GNN, no communication).

    Each agent has its own Q-network (shared weights) but no spatial aggregation.
    Used as baseline comparison for GAT-Double DQN.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        num_agents: int,
        lr: float = 3e-4,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.997,
        batch_size: int = 64,
        buffer_size: int = 50000,
        target_update_freq: int = 1000,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.num_agents = num_agents
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Q-Network (shared weights, no GAT)
        self.q_network = QNetwork(
            embed_dim=obs_dim,  # directly from observation
            num_actions=num_actions,
            hidden_dims=[128, 64],
        ).to(self.device)

        self.target_q_network = copy.deepcopy(self.q_network).to(self.device)
        self.target_q_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        self._train_steps = 0

    def select_actions(
        self, observations: np.ndarray, evaluate: bool = False
    ) -> np.ndarray:
        if not evaluate and np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions, size=self.num_agents)

        with torch.no_grad():
            obs_tensor = torch.tensor(observations, dtype=torch.float32).to(self.device)
            q_values = self.q_network(obs_tensor)
            actions = q_values.argmax(dim=-1).cpu().numpy()

        return actions

    def store_transition(self, states, actions, rewards, next_states, done):
        self.replay_buffer.push(states, actions, rewards, next_states, done)

    def train_step(self) -> Optional[Dict[str, float]]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Reshape: [batch, agents, dim] -> [batch*agents, dim]
        B, N = states_t.shape[0], states_t.shape[1]
        states_flat = states_t.reshape(B * N, -1)
        actions_flat = actions_t.reshape(B * N)
        rewards_flat = rewards_t.reshape(B * N)
        next_states_flat = next_states_t.reshape(B * N, -1)
        dones_flat = dones_t.unsqueeze(1).repeat(1, N).reshape(B * N)

        # Q-values
        q_values = self.q_network(states_flat)
        q_taken = q_values.gather(1, actions_flat.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            next_q_online = self.q_network(next_states_flat)
            best_actions = next_q_online.argmax(dim=-1)
            next_q_target = self.target_q_network(next_states_flat)
            next_q_best = next_q_target.gather(
                1, best_actions.unsqueeze(1)
            ).squeeze(1)
            target = rewards_flat + self.gamma * next_q_best * (1 - dones_flat)

        loss = nn.functional.mse_loss(q_taken, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        return {"loss_rl": loss.item()}

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str, extra_state: Optional[Dict[str, Any]] = None):
        checkpoint = {
            "q_network": self.q_network.state_dict(),
            "target_q_network": self.target_q_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "train_steps": self._train_steps,
            "replay_buffer": self.replay_buffer.state_dict(),
        }
        if extra_state:
            checkpoint.update(extra_state)
        torch.save(checkpoint, path)

    def load(self, path: str) -> Dict[str, Any]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_q_network.load_state_dict(checkpoint["target_q_network"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            _move_optimizer_state_to_device(self.optimizer, self.device)
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self._train_steps = checkpoint.get("train_steps", self._train_steps)
        if "replay_buffer" in checkpoint:
            self.replay_buffer.load_state_dict(checkpoint["replay_buffer"])

        return checkpoint
