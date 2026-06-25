"""
Evaluation script for trained models.
    python experiments/evaluate.py --model logs/gat_dqn_grid_2x2_.../best_model.pt --scenario grid_2x2 --agent gat_dqn
"""

import os
import sys
import argparse
import copy
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.envs.sumo_env import SumoEnvironment
from src.agents.dqn_agent import GATDoubleDQNAgent, IndependentDQNAgent
from src.utils.seeding import set_global_seed
from configs.default_config import DEFAULT_CONFIG, SCENARIOS


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


def _load_config_for_model(model_path: str) -> dict:
    """Load the training config next to a checkpoint, falling back to defaults."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    config_path = os.path.join(os.path.dirname(os.path.abspath(model_path)), "config.json")
    if not os.path.exists(config_path):
        return config

    with open(config_path, "r") as f:
        saved = json.load(f)

    if isinstance(saved, dict) and isinstance(saved.get("config"), dict):
        return saved["config"]
    return config


def evaluate(
    model_path: str,
    scenario_name: str = "grid_2x2",
    agent_type: str = "gat_dqn",
    num_episodes: int = 10,
    use_gui: bool = False,
    seed: int = 123,
    yellow_time: int = None,
    min_green: int = None,
    time_to_teleport: int = None,
):
    config = _load_config_for_model(model_path)
    if yellow_time is not None:
        config["env"]["yellow_time"] = yellow_time
    if min_green is not None:
        config["env"]["min_green"] = min_green
    if time_to_teleport is not None:
        config["env"]["time_to_teleport"] = time_to_teleport
    scenario = SCENARIOS[scenario_name]
    deterministic = config["training"].get("deterministic", True)

    # Global seeding for reproducibility across python/numpy/torch.
    set_global_seed(seed, deterministic=deterministic)

    net_file, route_file, sumocfg_file = _resolve_scenario_paths(scenario)

    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        sumocfg_file=sumocfg_file,
        use_gui=use_gui,
        seed=seed,
        lateral_resolution=scenario.get("lateral_resolution", 0.0),
        eff_vehicle_length=scenario.get("eff_vehicle_length", 0.0),
        **config["env"],
    )

    obs = env.reset()
    obs_dim = env.get_obs_size(env.ts_ids[0])
    num_actions = env.get_action_size(env.ts_ids[0])
    num_lanes = max(len(env.controlled_lanes[ts]) for ts in env.ts_ids)
    env.close()

    # Create agent and load model
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
        )
        agent.set_num_lanes(num_lanes)
    else:
        agent = IndependentDQNAgent(
            obs_dim=obs_dim,
            num_actions=num_actions,
            num_agents=env.num_agents,
            q_hidden_dims=config["q_network"]["hidden_dims"],
        )

    agent.load(model_path)

    # Run evaluation episodes
    all_rewards = []
    all_delays = []
    all_queues = []
    all_throughputs = []
    all_emergency_stops = []
    all_teleport_started = []
    all_teleport_ended = []
    all_episode_mean_delays = []
    all_episode_max_delays = []
    all_episode_mean_queues = []
    all_episode_max_queues = []

    for ep in range(1, num_episodes + 1):
        env.seed = seed + ep
        obs = env.reset()
        obs_array = np.stack([obs[ts_id] for ts_id in env.ts_ids])
        ep_reward = 0.0
        episode_step_metrics = []

        while True:
            actions_array = agent.select_actions(obs_array, evaluate=True)
            actions_dict = {
                ts_id: int(actions_array[i]) for i, ts_id in enumerate(env.ts_ids)
            }
            next_obs, rewards, done, info = env.step(actions_dict)
            episode_step_metrics.append(info["metrics"])
            next_obs_array = np.stack([next_obs[ts_id] for ts_id in env.ts_ids])
            rewards_array = np.array([rewards[ts_id] for ts_id in env.ts_ids])

            ep_reward += rewards_array.mean()
            obs_array = next_obs_array

            if done:
                break

        all_rewards.append(ep_reward)
        all_delays.append(info["metrics"]["avg_delay"])
        all_queues.append(info["metrics"]["avg_queue"])
        all_throughputs.append(info["metrics"]["throughput"])
        all_emergency_stops.append(info["metrics"]["emergency_stops"])
        all_teleport_started.append(info["metrics"]["teleport_started"])
        all_teleport_ended.append(info["metrics"]["teleport_ended"])

        delay_values = np.array([m["avg_delay"] for m in episode_step_metrics])
        queue_values = np.array([m["avg_queue"] for m in episode_step_metrics])
        mean_delay = float(np.mean(delay_values))
        max_delay = float(np.max(delay_values))
        mean_queue = float(np.mean(queue_values))
        max_queue = float(np.max(queue_values))
        all_episode_mean_delays.append(mean_delay)
        all_episode_max_delays.append(max_delay)
        all_episode_mean_queues.append(mean_queue)
        all_episode_max_queues.append(max_queue)

        print(
            f"Episode {ep}: reward={ep_reward:.2f}, "
            f"delay={info['metrics']['avg_delay']:.2f}, "
            f"queue={info['metrics']['avg_queue']:.2f}, "
            f"mean_delay={mean_delay:.2f}, "
            f"max_delay={max_delay:.2f}, "
            f"mean_queue={mean_queue:.2f}, "
            f"max_queue={max_queue:.2f}, "
            f"throughput={info['metrics']['throughput']}, "
            f"emergency_stops={info['metrics']['emergency_stops']}, "
            f"teleport_started={info['metrics']['teleport_started']}"
        )

    env.close()

    print(f"\n{'='*50}")
    print(f"Evaluation Results ({num_episodes} episodes)")
    print(f"{'='*50}")
    print(f"Avg Reward:  {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Avg Delay:   {np.mean(all_delays):.2f} ± {np.std(all_delays):.2f}")
    print(f"Avg Queue:   {np.mean(all_queues):.2f} ± {np.std(all_queues):.2f}")
    print(
        "Mean Delay:  "
        f"{np.mean(all_episode_mean_delays):.2f} ± {np.std(all_episode_mean_delays):.2f}"
    )
    print(
        "Max Delay:   "
        f"{np.mean(all_episode_max_delays):.2f} ± {np.std(all_episode_max_delays):.2f}"
    )
    print(
        "Mean Queue:  "
        f"{np.mean(all_episode_mean_queues):.2f} ± {np.std(all_episode_mean_queues):.2f}"
    )
    print(
        "Max Queue:   "
        f"{np.mean(all_episode_max_queues):.2f} ± {np.std(all_episode_max_queues):.2f}"
    )
    print(f"Throughput:  {np.mean(all_throughputs):.2f} ± {np.std(all_throughputs):.2f}")
    print(
        "Emergency:   "
        f"{np.mean(all_emergency_stops):.2f} ± {np.std(all_emergency_stops):.2f}"
    )
    print(
        "Teleport:    "
        f"{np.mean(all_teleport_started):.2f} ± {np.std(all_teleport_started):.2f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--scenario", type=str, default="grid_2x2", choices=list(SCENARIOS.keys()))
    parser.add_argument("--agent", type=str, default="gat_dqn", choices=["gat_dqn", "independent_dqn"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--yellow-time", type=int, default=None, help="Override env.yellow_time")
    parser.add_argument("--min-green", type=int, default=None, help="Override env.min_green")
    parser.add_argument("--time-to-teleport", type=int, default=None, help="Override env.time_to_teleport")

    args = parser.parse_args()
    evaluate(
        args.model,
        args.scenario,
        args.agent,
        args.episodes,
        args.gui,
        args.seed,
        args.yellow_time,
        args.min_green,
        args.time_to_teleport,
    )
