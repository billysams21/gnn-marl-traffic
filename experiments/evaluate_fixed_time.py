"""
Evaluate SUMO's fixed-time/default traffic-light program without training.

This baseline leaves the traffic-light logic controlled by SUMO and computes the
same reward definition used by the RL environment:
    episode_reward = sum_t mean_i(-(queue_i + alpha * waiting_i))
"""

import argparse
import csv
import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.envs.sumo_env import SumoEnvironment
from src.utils.seeding import set_global_seed
from configs.default_config import DEFAULT_CONFIG, SCENARIOS


def _resolve_scenario_paths(scenario: dict):
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


def evaluate_fixed_time(
    scenario_name: str,
    episodes: int,
    seed: int,
    use_gui: bool,
    net_file_override: str = "",
    route_file_override: str = "",
    sumocfg_file_override: str = "",
    yellow_time: int = None,
    min_green: int = None,
    recovery_enabled: bool = False,
    exp_suffix: str = "",
):
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if yellow_time is not None:
        config["env"]["yellow_time"] = yellow_time
    if min_green is not None:
        config["env"]["min_green"] = min_green
    config["env"]["recovery_enabled"] = recovery_enabled

    deterministic = config["training"].get("deterministic", True)
    set_global_seed(seed, deterministic=deterministic)

    scenario = SCENARIOS[scenario_name]
    net_file, route_file, sumocfg_file = _resolve_scenario_paths(scenario)
    if net_file_override:
        net_file = os.path.abspath(net_file_override)
    if route_file_override:
        route_file = os.path.abspath(route_file_override)
        sumocfg_file = None
    if sumocfg_file_override:
        sumocfg_file = os.path.abspath(sumocfg_file_override)
        route_file = None
    if route_file is None:
        raise ValueError("Completion rate requires an explicit route file.")
    scheduled_vehicles = _count_scheduled_vehicles(route_file)
    if scheduled_vehicles <= 0:
        raise ValueError("Completion rate requires at least one scheduled vehicle.")

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

    suffix = f"_{exp_suffix}" if exp_suffix else ""
    exp_name = (
        f"fixed_time_{scenario_name}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
    )
    log_dir = os.path.join(PROJECT_ROOT, "logs", exp_name)
    os.makedirs(log_dir, exist_ok=True)

    config_payload = {
        "config": config,
        "scenario": scenario,
        "run": {
            "agent_type": "fixed_time",
            "scenario_name": scenario_name,
            "seed": seed,
            "num_episodes_target": episodes,
            "use_gui": use_gui,
            "deterministic": deterministic,
            "exp_suffix": exp_suffix if exp_suffix else None,
            "env_overrides": {
                "yellow_time": yellow_time,
                "min_green": min_green,
                "recovery_enabled": recovery_enabled,
            },
            "file_overrides": {
                "net_file": net_file_override if net_file_override else None,
                "route_file": route_file_override if route_file_override else None,
                "sumocfg_file": sumocfg_file_override if sumocfg_file_override else None,
            },
        },
    }
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config_payload, f, indent=2, default=str)

    rows = []
    for ep in range(1, episodes + 1):
        env.seed = seed + ep
        env.tripinfo_output = os.path.join(log_dir, f"tripinfo_episode_{ep:03d}.xml")
        obs = env.reset(preserve_default_program=True)
        episode_reward = 0.0
        steps = 0

        while True:
            obs, rewards, done, info = env.step({})
            rewards_array = np.array([rewards[ts_id] for ts_id in env.ts_ids])
            episode_reward += rewards_array.mean()
            steps += 1

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
            "seed": env.seed,
            "reward": episode_reward,
            "steps": steps,
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
            "episode": ep,
            "timestamp": datetime.now().isoformat(),
            "tripinfo_file": os.path.basename(env.tripinfo_output),
        }
        rows.append(row)
        print(
            f"Episode {ep}: reward={episode_reward:.2f}, "
            f"median_time_loss={median_time_loss:.2f}, "
            f"mean_peak_queue={metrics['mean_peak_queue_per_lane']:.2f}, "
            f"completion_rate={row['completion_rate']:.2f}%, "
            f"critical_lane={metrics['critical_lane_id']}"
        )

    env.close()

    csv_path = os.path.join(log_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'=' * 50}")
    print(f"Fixed-Time Results ({episodes} episodes)")
    print(f"{'=' * 50}")
    for key, label in (
        ("median_time_loss", "Median timeLoss (s/vehicle)"),
        ("mean_peak_queue_per_lane", "Mean peak queue (vehicles/lane)"),
        ("completion_rate", "Completion rate (%)"),
    ):
        print(f"{label}: {_format_median_iqr([row[key] for row in rows])}")
    print(f"Logs saved to: {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fixed-time SUMO baseline")
    parser.add_argument("--scenario", type=str, default="grid_3x3", choices=list(SCENARIOS.keys()))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--net-file", type=str, default="", help="Override scenario net file")
    parser.add_argument("--route-file", type=str, default="", help="Override scenario route file")
    parser.add_argument("--sumocfg-file", type=str, default="", help="Override scenario SUMO config")
    parser.add_argument("--yellow-time", type=int, default=None, help="Saved for config parity; SUMO default program controls timing")
    parser.add_argument("--min-green", type=int, default=None, help="Saved for config parity; SUMO default program controls timing")
    parser.add_argument("--recovery", action="store_true", help="Enable recovery mode; disabled by default for clean evaluation")
    parser.add_argument("--exp-suffix", type=str, default="")

    args = parser.parse_args()
    evaluate_fixed_time(
        scenario_name=args.scenario,
        episodes=args.episodes,
        seed=args.seed,
        use_gui=args.gui,
        net_file_override=args.net_file,
        route_file_override=args.route_file,
        sumocfg_file_override=args.sumocfg_file,
        yellow_time=args.yellow_time,
        min_green=args.min_green,
        recovery_enabled=args.recovery,
        exp_suffix=args.exp_suffix,
    )
