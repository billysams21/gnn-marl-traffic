"""
Generate a synthetic PKJI input JSON for the 3x3 grid.

This is useful for stress/probe experiments while actual field volumes are not
available yet. Because volumes are synthetic, do not label the output as an
observed Indonesian baseline.
"""

import argparse
import json
import random
from pathlib import Path


def scaled_randint(rng: random.Random, low: int, high: int, multiplier: float) -> int:
    return max(int(round(rng.randint(low, high) * multiplier)), 0)


def movement(
    rng: random.Random,
    movement_dir: str,
    phase_id: int,
    multiplier: float,
    vol_mp_range,
    vol_sm_range,
    vol_ks_range,
    is_protected: bool,
    ltor_allowed: bool,
    ltor_lane_width_m: float,
):
    return {
        "movement_dir": movement_dir,
        "vol_mp": scaled_randint(rng, *vol_mp_range, multiplier),
        "vol_sm": scaled_randint(rng, *vol_sm_range, multiplier),
        "vol_ks": scaled_randint(rng, *vol_ks_range, multiplier),
        "is_protected": is_protected,
        "ltor_allowed": ltor_allowed,
        "ltor_lane_width_m": ltor_lane_width_m,
        "phase_id": phase_id,
    }


def forge_3x3_grid(
    seed: int,
    volume_multiplier: float,
    output: str,
    width_m: float = 10.0,
    num_lanes: int = 2,
    distance_m: float = 200.0,
    progression_speed_kmh: float = 40.0,
):
    rng = random.Random(seed)
    rows = ["A", "B", "C"]
    cols = ["0", "1", "2"]

    settings = {
        "preferred_yellow_s": 3,
        "base_saturation_flow_per_lane": 1800,
        "clamp_cycle": True,
        "generator": {
            "seed": seed,
            "volume_multiplier": volume_multiplier,
            "note": "Synthetic volumes for probing only; replace with observed PKJI inputs for calibrated baseline.",
        },
    }

    intersections = []
    corridors = []

    for row in rows:
        for col in cols:
            node_id = f"{row}{col}"
            approaches = []

            for approach_name, phase_id in [
                ("Utara", 1),
                ("Selatan", 1),
                ("Timur", 2),
                ("Barat", 2),
            ]:
                approaches.append(
                    {
                        "name": approach_name,
                        "width_m": width_m,
                        "num_lanes": num_lanes,
                        "movements": [
                            movement(
                                rng,
                                "L",
                                phase_id,
                                volume_multiplier,
                                (100, 150),
                                (800, 1000),
                                (10, 30),
                                is_protected=True,
                                ltor_allowed=True,
                                ltor_lane_width_m=2.5,
                            ),
                            movement(
                                rng,
                                "T",
                                phase_id,
                                volume_multiplier,
                                (400, 500),
                                (1800, 2200),
                                (40, 60),
                                is_protected=True,
                                ltor_allowed=False,
                                ltor_lane_width_m=0.0,
                            ),
                            movement(
                                rng,
                                "R",
                                phase_id,
                                volume_multiplier,
                                (80, 120),
                                (400, 600),
                                (10, 20),
                                is_protected=False,
                                ltor_allowed=False,
                                ltor_lane_width_m=0.0,
                            ),
                        ],
                    }
                )

            intersections.append(
                {
                    "intersection_id": node_id,
                    "num_phases": 2,
                    "offset_s": 0,
                    "approaches": approaches,
                }
            )

    for row in rows:
        for col_idx in range(2):
            corridors.append(
                {
                    "from_intersection": f"{row}{col_idx}",
                    "to_intersection": f"{row}{col_idx + 1}",
                    "distance_m": distance_m,
                    "progression_speed_kmh": progression_speed_kmh,
                    "main_direction": "east_west",
                }
            )

    for col in cols:
        for row_idx in range(2):
            corridors.append(
                {
                    "from_intersection": f"{rows[row_idx]}{col}",
                    "to_intersection": f"{rows[row_idx + 1]}{col}",
                    "distance_m": distance_m,
                    "progression_speed_kmh": progression_speed_kmh,
                    "main_direction": "north_south",
                }
            )

    grid_data = {
        "settings": settings,
        "intersections": intersections,
        "corridors": corridors,
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(grid_data, f, indent=2)

    print(f"Wrote synthetic PKJI grid input: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic PKJI 3x3 input")
    parser.add_argument("--output", default="configs/pkji_grid_3x3_full.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--volume-multiplier", type=float, default=1.0)
    parser.add_argument("--width-m", type=float, default=10.0)
    parser.add_argument("--num-lanes", type=int, default=2)
    parser.add_argument("--distance-m", type=float, default=200.0)
    parser.add_argument("--progression-speed-kmh", type=float, default=40.0)
    args = parser.parse_args()

    forge_3x3_grid(
        seed=args.seed,
        volume_multiplier=args.volume_multiplier,
        output=args.output,
        width_m=args.width_m,
        num_lanes=args.num_lanes,
        distance_m=args.distance_m,
        progression_speed_kmh=args.progression_speed_kmh,
    )
