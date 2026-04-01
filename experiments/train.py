"""
Training script for GAT-Double DQN traffic signal control.
    python experiments/train.py --scenario grid_2x2 --agent gat_dqn
    python experiments/train.py --scenario grid_3x3 --agent independent_dqn
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.envs.sumo_env import SumoEnvironment
from src.agents.dqn_agent import GATDoubleDQNAgent, IndependentDQNAgent
from src.utils.logger import MetricLogger
from configs.default_config import DEFAULT_CONFIG, SCENARIOS


def train(
    scenario_name: str = "grid_2x2",
    agent_type: str = "gat_dqn",
    num_episodes: int = 200,
    use_gui: bool = False,
    seed: int = 42,
    device: str = "auto",
):
    config = DEFAULT_CONFIG.copy()
    scenario = SCENARIOS[scenario_name]

    # Resolve file paths
    net_file = os.path.join(PROJECT_ROOT, scenario["net_file"])
    route_file = os.path.join(PROJECT_ROOT, scenario["route_file"])

    print(f"=" * 60)
    print(f"Training {agent_type} on {scenario_name}")
    print(f"Net: {net_file}")
    print(f"Route: {route_file}")
    print(f"Episodes: {num_episodes}")
    print(f"=" * 60)

    # ---- Create Environment ----
    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=use_gui,
        seed=seed,
        **config["env"],
    )

    print(f"Number of agents (traffic lights): {env.num_agents}")
    print(f"Traffic light IDs: {env.ts_ids}")

    # Initial reset to discover obs/action dimensions
    obs = env.reset()
    first_ts = env.ts_ids[0]
    obs_dim = env.get_obs_size(first_ts)
    num_actions = env.get_action_size(first_ts)
    print(f"Observation dim: {obs_dim}")
    print(f"Number of actions: {num_actions}")
    print(f"Adjacency matrix:\n{env.adjacency_matrix}")
    env.close()

    # ---- Create Agent ----
    if agent_type == "gat_dqn":
        agent = GATDoubleDQNAgent(
            obs_dim=obs_dim,
            num_actions=num_actions,
            num_agents=env.num_agents,
            edge_index=env.edge_index,
            gat_hidden_dim=config["gat"]["hidden_dim"],
            gat_embed_dim=config["gat"]["embed_dim"],
            gat_num_heads=config["gat"]["num_heads"],
            gat_dropout=config["gat"]["dropout"],
            q_hidden_dims=config["q_network"]["hidden_dims"],
            pred_lambda=config["prediction"]["lambda"],
            action_embed_dim=config["prediction"]["action_embed_dim"],
            lr=config["rl"]["lr"],
            gamma=config["rl"]["gamma"],
            epsilon_start=config["rl"]["epsilon_start"],
            epsilon_end=config["rl"]["epsilon_end"],
            epsilon_decay=config["rl"]["epsilon_decay"],
            batch_size=config["rl"]["batch_size"],
            buffer_size=config["rl"]["buffer_size"],
            target_update_freq=config["rl"]["target_update_freq"],
            device=device,
        )
    elif agent_type == "independent_dqn":
        agent = IndependentDQNAgent(
            obs_dim=obs_dim,
            num_actions=num_actions,
            num_agents=env.num_agents,
            lr=config["rl"]["lr"],
            gamma=config["rl"]["gamma"],
            epsilon_start=config["rl"]["epsilon_start"],
            epsilon_end=config["rl"]["epsilon_end"],
            epsilon_decay=config["rl"]["epsilon_decay"],
            batch_size=config["rl"]["batch_size"],
            buffer_size=config["rl"]["buffer_size"],
            target_update_freq=config["rl"]["target_update_freq"],
            device=device,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # ---- Logger ----
    exp_name = f"{agent_type}_{scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = MetricLogger(
        log_dir=os.path.join(PROJECT_ROOT, "logs"),
        experiment_name=exp_name,
    )
    logger.save_config({"config": config, "scenario": scenario, "agent_type": agent_type})

    # ---- Training Loop ----
    best_reward = -float("inf")

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        episode_reward = 0.0
        episode_losses = []
        step = 0

        # Convert obs dict to array [num_agents, obs_dim]
        obs_array = np.stack([obs[ts_id] for ts_id in env.ts_ids])

        while True:
            # Select actions
            actions_array = agent.select_actions(obs_array)

            # Convert to dict
            actions_dict = {
                ts_id: int(actions_array[i]) for i, ts_id in enumerate(env.ts_ids)
            }

            # Step environment
            next_obs, rewards, done, info = env.step(actions_dict)

            # Convert to arrays
            next_obs_array = np.stack([next_obs[ts_id] for ts_id in env.ts_ids])
            rewards_array = np.array([rewards[ts_id] for ts_id in env.ts_ids])

            # Store transition
            agent.store_transition(obs_array, actions_array, rewards_array, next_obs_array, done)

            # Train
            loss_info = agent.train_step()
            if loss_info is not None:
                episode_losses.append(loss_info)

            episode_reward += rewards_array.mean()
            obs_array = next_obs_array
            step += 1

            if done:
                break

        # Decay epsilon
        agent.decay_epsilon()

        # Compute average losses
        avg_losses = {}
        if episode_losses:
            for key in episode_losses[0]:
                avg_losses[key] = np.mean([l[key] for l in episode_losses])

        # Log
        metrics = {
            "reward": episode_reward,
            "epsilon": agent.epsilon,
            "steps": step,
            "avg_delay": info["metrics"]["avg_delay"],
            "avg_queue": info["metrics"]["avg_queue"],
            **avg_losses,
        }
        logger.log_episode(episode, metrics)

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            save_path = os.path.join(logger.log_dir, "best_model.pt")
            agent.save(save_path)

        # Periodic save
        if episode % config["training"]["save_interval"] == 0:
            save_path = os.path.join(logger.log_dir, f"checkpoint_ep{episode}.pt")
            agent.save(save_path)

    # Save final model
    agent.save(os.path.join(logger.log_dir, "final_model.pt"))
    env.close()

    print(f"\nTraining complete! Best reward: {best_reward:.2f}")
    print(f"Logs saved to: {logger.log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN-MARL traffic signal control")
    parser.add_argument(
        "--scenario", type=str, default="grid_2x2",
        choices=list(SCENARIOS.keys()),
        help="Network scenario to use",
    )
    parser.add_argument(
        "--agent", type=str, default="gat_dqn",
        choices=["gat_dqn", "independent_dqn"],
        help="Agent type",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--gui", action="store_true", help="Use SUMO GUI")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    args = parser.parse_args()

    train(
        scenario_name=args.scenario,
        agent_type=args.agent,
        num_episodes=args.episodes,
        use_gui=args.gui,
        seed=args.seed,
        device=args.device,
    )
