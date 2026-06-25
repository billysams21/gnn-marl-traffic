"""
Hyperparameter configuration for GNN-MARL traffic signal control.
"""

DEFAULT_CONFIG = {
    "env": {
        "num_seconds": 3600,       # 1 hour simulation
        "delta_time": 5,           # decision interval (seconds)
        "yellow_time": 2,
        "min_green": 12,
        "max_green": 40,
        "time_to_teleport": -1,
        "recovery_enabled": True,
        "recovery_enter_delay": 30.0,
        "recovery_enter_queue": 3.0,
        "recovery_exit_delay": 10.0,
        "recovery_exit_queue": 1.5,
        "recovery_hold_seconds": 30,
        "local_safety_enabled": False,
        "local_safety_lane_queue": 25.0,
        "local_safety_tl_queue": 120.0,
        "local_safety_mode": "queue",
        "local_safety_downstream_weight": 1.0,
        "local_safety_downstream_block_queue": 40.0,
        "local_safety_downstream_block_occupancy": 0.8,
        "local_safety_downstream_block_penalty": 50.0,
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
        "mode": "full",            # locked default: 'full' (main), 'simplified' (ablation)
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
        "eval_interval": 10,       # greedy validation every N episodes
        "eval_episodes": 1,        # validation episodes per eval point
        "eval_seed_offset": 10000, # fixed validation seeds: train_seed + offset + k
        "save_interval": 50,       # save checkpoint every N episodes
        "log_interval": 1,         # log metrics every N episodes
        "deterministic": True,     # reproducible runs across python/numpy/torch
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
    "grid_3x3_dynamic": {
        "net_file": "data/networks/grid_3x3/grid_3x3_pkji_m1.net.xml",
        "route_file": "data/networks/grid_3x3/grid_3x3_dynamic.rou.xml",
        "description": "3x3 grid (9 TL) with moving bottlenecks and heavy cross-traffic",
        "lateral_resolution": 0.8,
        "eff_vehicle_length": 3.6,
    },
    "grid_3x3_pkji_m1": {
        "net_file": "data/networks/grid_3x3/grid_3x3_pkji_m1.net.xml",
        "route_file": "data/networks/grid_3x3/grid_3x3_pkji_m1.rou.xml",
        "description": "3x3 grid with PKJI calibrated fixed-time net and PKJI-aware synthetic demand m1",
    },
    "grid_3x3_pkji_m1p5": {
        "net_file": "data/networks/grid_3x3/grid_3x3_pkji_m1p5.net.xml",
        "route_file": "data/networks/grid_3x3/grid_3x3_pkji_m1p5.rou.xml",
        "description": "3x3 grid with PKJI calibrated fixed-time net and PKJI-aware synthetic demand m1.5",
    },
    "arterial_stable": {
        "net_file": "data/networks/arterial_3x3/arterial_fixed.net.xml",
        "route_file": "data/networks/arterial_3x3/arterial_stable.rou.xml",
        "lateral_resolution": 0.7,
        # Per-lane queue normalization: 35% car (4.5+1.0=5.5m) + 65% moto (2.0+0.5=2.5m)
        # eff_vehicle_length = 0.35*5.5 + 0.65*2.5 = 3.55m (Indonesia minGap)
        "eff_vehicle_length": 3.55,
        "description": "Arterial+collector network (9 TL), PKJI-calibrated, 6000 veh uniform demand (stable)",
    },
    "arterial_peak": {
        "net_file": "data/networks/arterial_3x3/arterial_fixed.net.xml",
        "route_file": "data/networks/arterial_3x3/arterial_peak.rou.xml",
        "lateral_resolution": 0.7,
        "eff_vehicle_length": 3.55,
        "description": "Arterial+collector network (9 TL), PKJI-calibrated, 9000 veh Gaussian peak demand",
    },
    "arterial_unbalanced": {
        "net_file": "data/networks/arterial_3x3/arterial_fixed.net.xml",
        "route_file": "data/networks/arterial_3x3/arterial_unbalanced.rou.xml",
        "lateral_resolution": 0.7,
        "eff_vehicle_length": 3.55,
        "description": "Arterial+collector network (9 TL), PKJI-calibrated, 6000 veh directional imbalance (E->W dominates 6x)",
    },
    "toronto_small": {
        "net_file": "data/networks/toronto_small/toronto_small.net.xml",
        "sumocfg_file": "data/networks/toronto_small/toronto_small.sumocfg",
        "description": (
            "Small TorontoSUMONetworks-derived scenario for external validation "
            "after grid_3x3 main experiments and baselines are available"
        ),
    },
}
