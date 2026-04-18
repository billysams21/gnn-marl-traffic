"""
Evaluation script for trained models.
    python experiments/evaluate.py --model logs/gat_dqn_grid_2x2_.../best_model.pt --scenario grid_2x2 --agent gat_dqn
"""

import os
import sys
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.envs.sumo_env import SumoEnvironment
from src.agents.dqn_agent import GATDoubleDQNAgent, IndependentDQNAgent
from src.utils.seeding import set_global_seed
from configs.default_config import DEFAULT_CONFIG, SCENARIOS


def evaluate(
    model_path: str,
    scenario_name: str = "grid_2x2",
    agent_type: str = "gat_dqn",
    num_episodes: int = 10,
    use_gui: bool = False,
    seed: int = 123,
):
    config = DEFAULT_CONFIG.copy()
    scenario = SCENARIOS[scenario_name]
    deterministic = config["training"].get("deterministic", True)

    # Global seeding for reproducibility across python/numpy/torch.
    set_global_seed(seed, deterministic=deterministic)

    net_file = os.path.join(PROJECT_ROOT, scenario["net_file"])
    route_file = os.path.join(PROJECT_ROOT, scenario["route_file"])

    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=use_gui,
        seed=seed,
        **config["env"],
    )

    obs = env.reset()
    obs_dim = env.get_obs_size(env.ts_ids[0])
    num_actions = env.get_action_size(env.ts_ids[0])
    num_lanes = len(env.controlled_lanes[env.ts_ids[0]])
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
            prediction_mode=config["prediction"].get("mode", "full"),
        )
        agent.set_num_lanes(num_lanes)
    else:
        agent = IndependentDQNAgent(
            obs_dim=obs_dim,
            num_actions=num_actions,
            num_agents=env.num_agents,
        )

    agent.load(model_path)

    # Run evaluation episodes
    all_rewards = []
    all_delays = []
    all_queues = []

    for ep in range(1, num_episodes + 1):
        env.seed = seed + ep
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

        all_rewards.append(ep_reward)
        all_delays.append(info["metrics"]["avg_delay"])
        all_queues.append(info["metrics"]["avg_queue"])

        print(
            f"Episode {ep}: reward={ep_reward:.2f}, "
            f"delay={info['metrics']['avg_delay']:.2f}, "
            f"queue={info['metrics']['avg_queue']:.2f}"
        )

    env.close()

    print(f"\n{'='*50}")
    print(f"Evaluation Results ({num_episodes} episodes)")
    print(f"{'='*50}")
    print(f"Avg Reward:  {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Avg Delay:   {np.mean(all_delays):.2f} ± {np.std(all_delays):.2f}")
    print(f"Avg Queue:   {np.mean(all_queues):.2f} ± {np.std(all_queues):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--scenario", type=str, default="grid_2x2", choices=list(SCENARIOS.keys()))
    parser.add_argument("--agent", type=str, default="gat_dqn", choices=["gat_dqn", "independent_dqn"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()
    evaluate(args.model, args.scenario, args.agent, args.episodes, args.gui, args.seed)
