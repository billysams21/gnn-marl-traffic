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


def _checkpoint_meta(
    episode: int,
    best_train_reward: float,
    best_eval_reward: float,
    best_model_metric: str,
):
    """Build checkpoint metadata with reproducibility state."""
    return {
        "episode": episode,
        # Keep the legacy key for old scripts while recording explicit criteria.
        "best_reward": (
            best_eval_reward if best_model_metric == "eval_reward" else best_train_reward
        ),
        "best_train_reward": best_train_reward,
        "best_eval_reward": best_eval_reward,
        "best_model_metric": best_model_metric,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def _run_greedy_validation(agent, env: SumoEnvironment, seed: int, episodes: int) -> dict:
    """Run greedy validation episodes without replay storage or epsilon exploration."""
    if episodes <= 0:
        return {}

    original_seed = env.seed
    rewards = []
    delays = []
    queues = []
    throughputs = []
    emergency_stops = []

    try:
        for i in range(episodes):
            env.seed = seed + i
            obs = env.reset()
            obs_array = np.stack([obs[ts_id] for ts_id in env.ts_ids])
            episode_reward = 0.0

            while True:
                actions_array = agent.select_actions(obs_array, evaluate=True)
                actions_dict = {
                    ts_id: int(actions_array[j]) for j, ts_id in enumerate(env.ts_ids)
                }
                next_obs, step_rewards, done, info = env.step(actions_dict)
                next_obs_array = np.stack([next_obs[ts_id] for ts_id in env.ts_ids])
                rewards_array = np.array([step_rewards[ts_id] for ts_id in env.ts_ids])

                episode_reward += rewards_array.mean()
                obs_array = next_obs_array

                if done:
                    break

            rewards.append(episode_reward)
            delays.append(info["metrics"]["avg_delay"])
            queues.append(info["metrics"]["avg_queue"])
            throughputs.append(info["metrics"]["throughput"])
            emergency_stops.append(info["metrics"]["emergency_stops"])
    finally:
        env.close()
        env.seed = original_seed

    return {
        "eval_reward": float(np.mean(rewards)),
        "eval_reward_std": float(np.std(rewards)),
        "eval_avg_delay": float(np.mean(delays)),
        "eval_avg_queue": float(np.mean(queues)),
        "eval_throughput": float(np.mean(throughputs)),
        "eval_emergency_stops": float(np.mean(emergency_stops)),
    }


def _resolve_scenario_paths(scenario: dict):
    """Resolve scenario files relative to the project root."""
    net_file = os.path.join(PROJECT_ROOT, scenario["net_file"])

    route_config = scenario.get("route_file")
    if isinstance(route_config, list):
        route_file = [os.path.join(PROJECT_ROOT, path) for path in route_config]
    elif route_config:
        route_file = os.path.join(PROJECT_ROOT, route_config)
    else:
        route_file = None

    sumocfg_config = scenario.get("sumocfg_file")
    sumocfg_file = (
        os.path.join(PROJECT_ROOT, sumocfg_config) if sumocfg_config else None
    )

    return net_file, route_file, sumocfg_file


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
    max_green: Optional[int] = None,
    time_to_teleport: Optional[int] = None,
    epsilon_start: Optional[float] = None,
    epsilon_end: Optional[float] = None,
    epsilon_decay: Optional[float] = None,
    lr: Optional[float] = None,
    aux_weight: Optional[float] = None,
    eval_interval: Optional[int] = None,
    eval_episodes: Optional[int] = None,
    exp_suffix: str = "",
):
    config = copy.deepcopy(DEFAULT_CONFIG)
    scenario = SCENARIOS[scenario_name]
    if yellow_time is not None:
        config["env"]["yellow_time"] = yellow_time
    if min_green is not None:
        config["env"]["min_green"] = min_green
    if max_green is not None:
        config["env"]["max_green"] = max_green
    if time_to_teleport is not None:
        config["env"]["time_to_teleport"] = time_to_teleport
    if epsilon_start is not None:
        config["rl"]["epsilon_start"] = epsilon_start
    if epsilon_end is not None:
        config["rl"]["epsilon_end"] = epsilon_end
    if epsilon_decay is not None:
        config["rl"]["epsilon_decay"] = epsilon_decay
    if lr is not None:
        config["rl"]["lr"] = lr
    if aux_weight is not None:
        config["prediction"]["lambda"] = aux_weight
    if eval_interval is not None:
        config["training"]["eval_interval"] = eval_interval
    if eval_episodes is not None:
        config["training"]["eval_episodes"] = eval_episodes
    deterministic = config["training"].get("deterministic", True)
    resume_mode = bool(resume_checkpoint)

    # Global seeding for reproducibility across python/numpy/torch.
    set_global_seed(seed, deterministic=deterministic)

    # Resolve file paths
    net_file, route_file, sumocfg_file = _resolve_scenario_paths(scenario)

    print(f"=" * 60)
    print(f"Training {agent_type} on {scenario_name}")
    print(f"Net: {net_file}")
    if sumocfg_file:
        print(f"SUMO config: {sumocfg_file}")
    else:
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
        sumocfg_file=sumocfg_file,
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
            q_hidden_dims=config["q_network"]["hidden_dims"],
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
        "training_overrides": {
            "eval_interval": eval_interval,
            "eval_episodes": eval_episodes,
        },
        "rl_overrides": {
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay": epsilon_decay,
            "lr": lr,
            "aux_weight": aux_weight,
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
    best_train_reward = -float("inf")
    best_eval_reward = -float("inf")
    best_model_metric = "eval_reward"
    eval_interval = int(config["training"].get("eval_interval", 0) or 0)
    eval_episodes = int(config["training"].get("eval_episodes", 0) or 0)
    eval_seed = seed + int(config["training"].get("eval_seed_offset", 10000))

    if resume_mode:
        checkpoint = agent.load(resume_checkpoint)
        resume_episode = checkpoint.get("episode", None)
        if resume_episode is None:
            resume_episode = _infer_episode_from_checkpoint_path(resume_checkpoint)
        start_episode = int(resume_episode) + 1
        best_train_reward = float(
            checkpoint.get(
                "best_train_reward",
                checkpoint.get("best_reward", best_train_reward),
            )
        )
        best_eval_reward = float(checkpoint.get("best_eval_reward", best_eval_reward))
        best_model_metric = checkpoint.get("best_model_metric", best_model_metric)

        rng_state = checkpoint.get("rng_state", None)
        if rng_state is not None:
            _restore_rng_state(rng_state)

        # Allow explicit epsilon override during resume
        if epsilon_start is not None:
            agent.epsilon = epsilon_start
            print(f"Overriding epsilon from checkpoint to {agent.epsilon:.4f}")
        if epsilon_end is not None:
            agent.epsilon_end = epsilon_end
            print(f"Overriding epsilon_end to {agent.epsilon_end:.4f}")
        if epsilon_decay is not None:
            agent.epsilon_decay = epsilon_decay
            print(f"Overriding epsilon_decay to {agent.epsilon_decay:.4f}")

        print(
            f"Resuming from episode {start_episode} "
            f"(best_train_reward={best_train_reward:.2f}, "
            f"best_eval_reward={best_eval_reward:.2f}, "
            f"replay_size={len(agent.replay_buffer)}, "
            f"epsilon={agent.epsilon:.4f})"
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

        eval_metrics = {}
        should_eval = (
            eval_interval > 0
            and eval_episodes > 0
            and episode % eval_interval == 0
        )
        if should_eval:
            eval_metrics = _run_greedy_validation(
                agent=agent,
                env=env,
                seed=eval_seed,
                episodes=eval_episodes,
            )
            metrics.update(eval_metrics)

        logger.log_episode(episode, metrics)

        # Save best model. Prefer greedy validation reward when available; fall back
        # to exploratory training reward only before the first validation point.
        if episode_reward > best_train_reward:
            best_train_reward = episode_reward

        should_save_best = False
        if eval_metrics:
            best_model_metric = "eval_reward"
            if eval_metrics["eval_reward"] > best_eval_reward:
                best_eval_reward = eval_metrics["eval_reward"]
                should_save_best = True
        elif best_eval_reward == -float("inf") and episode_reward >= best_train_reward:
            best_model_metric = "train_reward"
            should_save_best = True

        if should_save_best:
            checkpoint_meta = _checkpoint_meta(
                episode=episode,
                best_train_reward=best_train_reward,
                best_eval_reward=best_eval_reward,
                best_model_metric=best_model_metric,
            )
            save_path = os.path.join(logger.log_dir, "best_model.pt")
            agent.save(save_path, extra_state=checkpoint_meta)

        # Periodic save
        if episode % config["training"]["save_interval"] == 0:
            checkpoint_meta = _checkpoint_meta(
                episode=episode,
                best_train_reward=best_train_reward,
                best_eval_reward=best_eval_reward,
                best_model_metric=best_model_metric,
            )
            save_path = os.path.join(logger.log_dir, f"checkpoint_ep{episode}.pt")
            agent.save(save_path, extra_state=checkpoint_meta)

    # Save final model
    agent.save(
        os.path.join(logger.log_dir, "final_model.pt"),
        extra_state=_checkpoint_meta(
            episode=num_episodes,
            best_train_reward=best_train_reward,
            best_eval_reward=best_eval_reward,
            best_model_metric=best_model_metric,
        ),
    )
    env.close()

    print(
        f"\nTraining complete! "
        f"Best train reward: {best_train_reward:.2f} | "
        f"Best eval reward: {best_eval_reward:.2f}"
    )
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
    parser.add_argument("--max-green", type=int, default=None, help="Override env.max_green")
    parser.add_argument("--time-to-teleport", type=int, default=None, help="Override env.time_to_teleport")
    parser.add_argument("--epsilon-start", type=float, default=None, help="Override rl.epsilon_start")
    parser.add_argument("--epsilon-end", type=float, default=None, help="Override rl.epsilon_end")
    parser.add_argument("--epsilon-decay", type=float, default=None, help="Override rl.epsilon_decay")
    parser.add_argument("--lr", type=float, default=None, help="Override rl.lr")
    parser.add_argument(
        "--aux-weight",
        type=float,
        default=None,
        help="Override prediction.lambda (auxiliary loss weight)",
    )
    parser.add_argument("--eval-interval", type=int, default=None, help="Override training.eval_interval")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Override training.eval_episodes")
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
        max_green=args.max_green,
        time_to_teleport=args.time_to_teleport,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        lr=args.lr,
        aux_weight=args.aux_weight,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        exp_suffix=args.exp_suffix,
    )
