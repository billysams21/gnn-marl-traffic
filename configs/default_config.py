"""
Hyperparameter configuration for GNN-MARL traffic signal control.
"""

DEFAULT_CONFIG = {
    "env": {
        "num_seconds": 3600,       # 1 hour simulation
        "delta_time": 5,           # decision interval (seconds)
        "yellow_time": 3,
        "min_green": 10,
        "max_green": 60,
        "reward_alpha": 0.5,       # r = -(queue + alpha * waiting)
        # Normalization parameters (best practice)
        "max_queue_per_lane": 30,      # Max vehicles per lane
        "max_waiting_time": 300.0,     # Max waiting time (5 min)
        "max_delta_queue": 10.0,       # Max queue change per step
        "max_delta_density": 0.5,      # Max density change per step
    },

    "gat": {
        "hidden_dim": 64,
        "embed_dim": 64,
        "num_heads": 4,
        "dropout": 0.0,
    },

    "q_network": {
        "hidden_dims": [128, 64],
    },

    "prediction": {
        "lambda": 0.3,             # L_total = L_RL + lambda * L_pred
        "action_embed_dim": 8,
        "mode": "simplified",       # 'simplified' (proposal) or 'full'
    },

    "rl": {
        "lr": 3e-4,
        "gamma": 0.95,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.997,
        "batch_size": 64,
        "buffer_size": 50000,
        "target_update_freq": 1000,
    },

    "training": {
        "num_episodes": 200,
        "eval_interval": 10,       # evaluate every N episodes
        "save_interval": 50,       # save checkpoint every N episodes
        "log_interval": 1,         # log metrics every N episodes
    },
}


SCENARIOS = {
    "grid_2x2": {
        "net_file": "data/networks/grid_2x2/grid_2x2.net.xml",
        "route_file": "data/networks/grid_2x2/grid_2x2.rou.xml",
        "description": "2x2 grid (4 intersections) for development",
    },
    "grid_3x3": {
        "net_file": "data/networks/grid_3x3/grid_3x3.net.xml",
        "route_file": "data/networks/grid_3x3/grid_3x3.rou.xml",
        "description": "3x3 grid (9 intersections) for main experiments",
    },
}
