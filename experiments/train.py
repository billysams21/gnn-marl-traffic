"""
Training script for GAT-Double DQN traffic signal control.
    python experiments/train.py --scenario grid_2x2 --agent gat_dqn
    python experiments/train.py --scenario grid_3x3 --agent independent_dqn
"""

import os
import sys
import argparse
import copy
import random
import re
import numpy as np
import torch
from datetime import datetime
from typing import Optional

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.envs.sumo_env import SumoEnvironment
from src.agents.dqn_agent import GATDoubleDQNAgent, IndependentDQNAgent
from src.utils.logger import MetricLogger
from src.utils.seeding import set_global_seed
from configs.default_config import DEFAULT_CONFIG, SCENARIOS


def _restore_rng_state(rng_state: dict):
    """Restore python/numpy/torch RNG states from checkpoint safely."""
    if not rng_state:
        return

    if "python" in rng_state and rng_state["python"] is not None:
        random.setstate(rng_state["python"])

    if "numpy" in rng_state and rng_state["numpy"] is not None:
        np_state = rng_state["numpy"]
        # numpy state must be tuple(name, keys[uint32], pos, has_gauss, cached_gaussian)
        if isinstance(np_state, tuple) and len(np_state) == 5:
            keys = np.asarray(np_state[1], dtype=np.uint32)
            np.random.set_state((np_state[0], keys, np_state[2], np_state[3], np_state[4]))

    if "torch" in rng_state and rng_state["torch"] is not None:
        torch_state = rng_state["torch"]
        if not isinstance(torch_state, torch.ByteTensor):
            torch_state = torch.as_tensor(torch_state, dtype=torch.uint8)
        torch.set_rng_state(torch_state.cpu())

    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        cuda_state = rng_state["cuda"]
        try:
            # Expected format: list[ByteTensor], one state per visible CUDA device.
            normalized = []
            for s in cuda_state:
                if not isinstance(s, torch.ByteTensor):
                    s = torch.as_tensor(s, dtype=torch.uint8)
                normalized.append(s.cpu())
            torch.cuda.set_rng_state_all(normalized)
        except Exception as exc:
            print(f"Warning: skipped CUDA RNG restore due to format mismatch: {exc}")


def _infer_episode_from_checkpoint_path(checkpoint_path: str) -> int:
    """Fallback for old checkpoints without explicit episode metadata."""
    match = re.search(r"checkpoint_ep(\d+)\.pt$", os.path.basename(checkpoint_path))
    if match:
        return int(match.group(1))
    return 0


def train(
    scenario_name: str = "grid_2x2",
    agent_type: str = "gat_dqn",
    num_episodes: int = 200,
    use_gui: bool = False,
    seed: int = 42,
    device: str = "auto",
    resume_checkpoint: str = "",
    resume_log_dir: str = "",
    yellow_time: Optional[int] = None,
    min_green: Optional[int] = None,
    exp_suffix: str = "",
):
    config = copy.deepcopy(DEFAULT_CONFIG)
    scenario = SCENARIOS[scenario_name]
    if yellow_time is not None:
        config["env"]["yellow_time"] = yellow_time
    if min_green is not None:
        config["env"]["min_green"] = min_green
    deterministic = config["training"].get("deterministic", True)
    resume_mode = bool(resume_checkpoint)

    # Global seeding for reproducibility across python/numpy/torch.
    set_global_seed(seed, deterministic=deterministic)

    # Resolve file paths
    net_file = os.path.join(PROJECT_ROOT, scenario["net_file"])
    route_file = os.path.join(PROJECT_ROOT, scenario["route_file"])

    print(f"=" * 60)
    print(f"Training {agent_type} on {scenario_name}")
    print(f"Net: {net_file}")
    print(f"Route: {route_file}")
    print(f"Episodes: {num_episodes}")
    print(f"Seed: {seed} | Deterministic: {deterministic}")
    if resume_mode:
        print(f"Resume checkpoint: {resume_checkpoint}")
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
    num_lanes = len(env.controlled_lanes[first_ts])
    print(f"Observation dim: {obs_dim}")
    print(f"Number of actions: {num_actions}")
    print(f"Number of lanes per intersection: {num_lanes}")
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
            prediction_mode=config["prediction"].get("mode", "full"),
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
        # Set number of lanes for simplified prediction head
        agent.set_num_lanes(num_lanes)
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
    if resume_mode:
        resolved_resume_log_dir = resume_log_dir or os.path.dirname(
            os.path.abspath(resume_checkpoint)
        )
        if not os.path.isabs(resolved_resume_log_dir):
            resolved_resume_log_dir = os.path.join(PROJECT_ROOT, resolved_resume_log_dir)
        logger = MetricLogger(log_dir=resolved_resume_log_dir, experiment_name=None)
    else:
        suffix = f"_{exp_suffix}" if exp_suffix else ""
        exp_name = f"{agent_type}_{scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
        logger = MetricLogger(
            log_dir=os.path.join(PROJECT_ROOT, "logs"),
            experiment_name=exp_name,
        )

    run_metadata = {
        "agent_type": agent_type,
        "scenario_name": scenario_name,
        "seed": seed,
        "device": str(agent.device),
        "num_episodes_target": num_episodes,
        "use_gui": use_gui,
        "deterministic": deterministic,
        "resume_mode": resume_mode,
        "resume_checkpoint": resume_checkpoint if resume_mode else None,
        "resume_log_dir": resume_log_dir if resume_mode else None,
        "exp_suffix": exp_suffix if exp_suffix else None,
        "env_overrides": {
            "yellow_time": yellow_time,
            "min_green": min_green,
        },
    }
    logger.save_config(
        {
            "config": config,
            "scenario": scenario,
            "run": run_metadata,
        },
        overwrite=not resume_mode,
    )

    # ---- Training Loop ----
    start_episode = 1
    best_reward = -float("inf")

    if resume_mode:
        checkpoint = agent.load(resume_checkpoint)
        resume_episode = checkpoint.get("episode", None)
        if resume_episode is None:
            resume_episode = _infer_episode_from_checkpoint_path(resume_checkpoint)
        start_episode = int(resume_episode) + 1
        best_reward = float(checkpoint.get("best_reward", best_reward))

        rng_state = checkpoint.get("rng_state", None)
        if rng_state is not None:
            _restore_rng_state(rng_state)

        print(
            f"Resuming from episode {start_episode} "
            f"(best_reward={best_reward:.2f}, replay_size={len(agent.replay_buffer)})"
        )

    if start_episode > num_episodes:
        env.close()
        print(
            f"Checkpoint already passed requested episodes "
            f"(start={start_episode}, requested={num_episodes}). Nothing to do."
        )
        return

    for episode in range(start_episode, num_episodes + 1):
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
            "throughput": info["metrics"]["throughput"],
            "emergency_stops": info["metrics"]["emergency_stops"],
            **avg_losses,
        }
        logger.log_episode(episode, metrics)

        checkpoint_meta = {
            "episode": episode,
            "best_reward": best_reward,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            checkpoint_meta["best_reward"] = best_reward
            save_path = os.path.join(logger.log_dir, "best_model.pt")
            agent.save(save_path, extra_state=checkpoint_meta)

        # Periodic save
        if episode % config["training"]["save_interval"] == 0:
            save_path = os.path.join(logger.log_dir, f"checkpoint_ep{episode}.pt")
            agent.save(save_path, extra_state=checkpoint_meta)

    # Save final model
    agent.save(
        os.path.join(logger.log_dir, "final_model.pt"),
        extra_state={
            "episode": num_episodes,
            "best_reward": best_reward,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
    )
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
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default="",
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--resume-log-dir",
        type=str,
        default="",
        help="Optional existing experiment log directory for resumed metrics/config",
    )
    parser.add_argument("--yellow-time", type=int, default=None, help="Override env.yellow_time")
    parser.add_argument("--min-green", type=int, default=None, help="Override env.min_green")
    parser.add_argument("--exp-suffix", type=str, default="", help="Suffix for experiment folder name")

    args = parser.parse_args()

    train(
        scenario_name=args.scenario,
        agent_type=args.agent,
        num_episodes=args.episodes,
        use_gui=args.gui,
        seed=args.seed,
        device=args.device,
        resume_checkpoint=args.resume_checkpoint,
        resume_log_dir=args.resume_log_dir,
        yellow_time=args.yellow_time,
        min_green=args.min_green,
        exp_suffix=args.exp_suffix,
    )
