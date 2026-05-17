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
    exp_suffix: str = "",
):
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if yellow_time is not None:
        config["env"]["yellow_time"] = yellow_time
    if min_green is not None:
        config["env"]["min_green"] = min_green

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

    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        sumocfg_file=sumocfg_file,
        use_gui=use_gui,
        seed=seed,
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

        row = {
            "reward": episode_reward,
            "steps": steps,
            "avg_delay": info["metrics"]["avg_delay"],
            "avg_queue": info["metrics"]["avg_queue"],
            "throughput": info["metrics"]["throughput"],
            "emergency_stops": info["metrics"]["emergency_stops"],
            "episode": ep,
            "timestamp": datetime.now().isoformat(),
        }
        rows.append(row)
        print(
            f"Episode {ep}: reward={episode_reward:.2f}, "
            f"delay={row['avg_delay']:.2f}, queue={row['avg_queue']:.2f}, "
            f"throughput={row['throughput']}, emergency_stops={row['emergency_stops']}"
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
    print(f"Avg Reward:  {np.mean([r['reward'] for r in rows]):.2f} ± {np.std([r['reward'] for r in rows]):.2f}")
    print(f"Avg Delay:   {np.mean([r['avg_delay'] for r in rows]):.2f} ± {np.std([r['avg_delay'] for r in rows]):.2f}")
    print(f"Avg Queue:   {np.mean([r['avg_queue'] for r in rows]):.2f} ± {np.std([r['avg_queue'] for r in rows]):.2f}")
    print(f"Throughput:  {np.mean([r['throughput'] for r in rows]):.2f} ± {np.std([r['throughput'] for r in rows]):.2f}")
    print(f"Emergency:   {np.mean([r['emergency_stops'] for r in rows]):.2f} ± {np.std([r['emergency_stops'] for r in rows]):.2f}")
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
        exp_suffix=args.exp_suffix,
    )
