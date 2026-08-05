"""
Evaluation script for trained models.
    python experiments/evaluate.py --model logs/gat_dqn_grid_2x2_.../best_model.pt --scenario grid_2x2 --agent gat_dqn
"""

import os
import sys
import argparse
import copy
import csv
import json
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime

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


def _count_scheduled_vehicles(route_file) -> int:
    """Count explicitly scheduled vehicles and trips in route files used for evaluation."""
    route_files = route_file if isinstance(route_file, list) else [route_file]
    total = 0
    for path in route_files:
        if not path:
            continue
        root = ET.parse(path).getroot()
        total += len(root.findall("vehicle")) + len(root.findall("trip"))
        for flow in root.findall("flow"):
            if "number" not in flow.attrib:
                raise ValueError(
                    "Completion rate requires explicit vehicle/trip demand or "
                    f"a flow 'number' attribute: {path}"
                )
            total += int(float(flow.attrib["number"]))
    return total


def _tripinfo_metrics(tripinfo_path: str) -> tuple[float, float, int]:
    """Return mean and median SUMO timeLoss for vehicles that arrived."""
    root = ET.parse(tripinfo_path).getroot()
    time_losses = np.array(
        [float(trip.attrib["timeLoss"]) for trip in root.findall("tripinfo")],
        dtype=float,
    )
    if not len(time_losses):
        return float("nan"), float("nan"), 0
    return float(np.mean(time_losses)), float(np.median(time_losses)), len(time_losses)


def _format_median_iqr(values) -> str:
    values = np.asarray(values, dtype=float)
    return (
        f"{np.median(values):.2f} "
        f"[{np.percentile(values, 25):.2f}, {np.percentile(values, 75):.2f}]"
    )


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
    recovery_enabled: bool = None,
    output_dir: str = None,
):
    config = _load_config_for_model(model_path)
    if yellow_time is not None:
        config["env"]["yellow_time"] = yellow_time
    if min_green is not None:
        config["env"]["min_green"] = min_green
    if time_to_teleport is not None:
        config["env"]["time_to_teleport"] = time_to_teleport
    if recovery_enabled is not None:
        config["env"]["recovery_enabled"] = recovery_enabled
    scenario = SCENARIOS[scenario_name]
    deterministic = config["training"].get("deterministic", True)

    print(
        "Evaluation env overrides: "
        f"time_to_teleport={config['env'].get('time_to_teleport')}, "
        f"recovery_enabled={config['env'].get('recovery_enabled')}, "
        f"yellow_time={config['env'].get('yellow_time')}, "
        f"min_green={config['env'].get('min_green')}"
    )

    # Global seeding for reproducibility across python/numpy/torch.
    set_global_seed(seed, deterministic=deterministic)

    net_file, route_file, sumocfg_file = _resolve_scenario_paths(scenario)
    scheduled_vehicles = _count_scheduled_vehicles(route_file)
    if scheduled_vehicles <= 0:
        raise ValueError("Completion rate requires at least one scheduled vehicle.")

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(model_path)),
            "evaluations",
            datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    os.makedirs(output_dir, exist_ok=True)

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
    rows = []

    for ep in range(1, num_episodes + 1):
        env.seed = seed + ep
        env.tripinfo_output = os.path.join(output_dir, f"tripinfo_episode_{ep:03d}.xml")
        obs = env.reset()
        obs_array = np.stack([obs[ts_id] for ts_id in env.ts_ids])
        ep_reward = 0.0

        while True:
            actions_array = agent.select_actions(obs_array, evaluate=True)
            actions_dict = {
                ts_id: int(actions_array[i]) for i, ts_id in enumerate(env.ts_ids)
            }
            next_obs, rewards, done, info = env.step(actions_dict)
            next_obs_array = np.stack([next_obs[ts_id] for ts_id in env.ts_ids])
            rewards_array = np.array([rewards[ts_id] for ts_id in env.ts_ids])

            ep_reward += rewards_array.mean()
            obs_array = next_obs_array

            if done:
                break

        mean_time_loss, median_time_loss, tripinfo_count = _tripinfo_metrics(
            env.tripinfo_output
        )
        metrics = info["metrics"]
        if tripinfo_count != metrics["throughput"]:
            raise RuntimeError(
                f"tripinfo count ({tripinfo_count}) does not match TraCI throughput "
                f"({metrics['throughput']}) in episode {ep}."
            )
        row = {
            "episode": ep,
            "seed": env.seed,
            "reward": ep_reward,
            "mean_time_loss": mean_time_loss,
            "median_time_loss": median_time_loss,
            "completion_rate": 100.0 * tripinfo_count / scheduled_vehicles,
            "throughput": tripinfo_count,
            "scheduled_vehicles": scheduled_vehicles,
            "mean_peak_queue_per_lane": metrics["mean_peak_queue_per_lane"],
            "max_peak_queue": metrics["max_peak_queue"],
            "max_peak_queue_occupancy_ratio": metrics["max_peak_queue_occupancy_ratio"],
            "critical_lane_id": metrics["critical_lane_id"],
            "emergency_stops": metrics["emergency_stops"],
            "teleport_started": metrics["teleport_started"],
            "teleport_ended": metrics["teleport_ended"],
            "tripinfo_file": os.path.basename(env.tripinfo_output),
        }
        rows.append(row)

        print(
            f"Episode {ep}: reward={ep_reward:.2f}, "
            f"median_time_loss={median_time_loss:.2f}, "
            f"mean_peak_queue={metrics['mean_peak_queue_per_lane']:.2f}, "
            f"completion_rate={row['completion_rate']:.2f}%, "
            f"critical_lane={metrics['critical_lane_id']}"
        )

    env.close()

    print(f"\n{'='*50}")
    print(f"Evaluation Results ({num_episodes} episodes)")
    print(f"{'='*50}")
    for key, label in (
        ("median_time_loss", "Median timeLoss (s/vehicle)"),
        ("mean_peak_queue_per_lane", "Mean peak queue (vehicles/lane)"),
        ("completion_rate", "Completion rate (%)"),
    ):
        print(f"{label}: {_format_median_iqr([row[key] for row in rows])}")
    csv_path = os.path.join(output_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Logs saved to: {output_dir}")


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
    parser.add_argument("--no-recovery", action="store_true", help="Disable env.recovery_enabled for clean evaluation")
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()
    recovery_enabled = False if args.no_recovery else None
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
        recovery_enabled,
        args.output_dir,
    )
