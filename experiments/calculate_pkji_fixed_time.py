"""
Calculate PKJI-style fixed-time signal plans from a JSON input file.

Example:
    python experiments/calculate_pkji_fixed_time.py \
        --input configs/pkji_grid_3x3_template.json \
        --output logs/pkji_plan.json
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.baselines.pkji import (
    CorridorLink,
    Intersection,
    calculate_intersection_plan,
    calculate_offsets_from_corridors,
    plan_to_dict,
)


def load_inputs(path: str):
    with open(path, "r") as f:
        payload = json.load(f)

    intersections = [
        Intersection.from_dict(item) for item in payload.get("intersections", [])
    ]
    corridors = [CorridorLink.from_dict(item) for item in payload.get("corridors", [])]
    settings = payload.get("settings", {})
    return intersections, corridors, settings


def write_summary_csv(path: str, plans):
    rows = []
    for plan in plans:
        for phase in plan.phases:
            rows.append(
                {
                    "intersection_id": plan.intersection_id,
                    "phase_id": phase.phase_id,
                    "cycle_s": round(plan.cycle_s, 3),
                    "offset_s": round(plan.offset_s, 3),
                    "total_lost_time_s": round(plan.total_lost_time_s, 3),
                    "sum_critical_ratio": round(plan.sum_critical_ratio, 6),
                    "oversaturated": plan.oversaturated,
                    "cycle_clamped": plan.cycle_clamped,
                    "critical_ratio": round(phase.critical_ratio, 6),
                    "green_s": round(phase.green_s, 3),
                    "intergreen_s": round(phase.intergreen_s, 3),
                    "yellow_s": round(phase.yellow_s, 3),
                    "all_red_s": round(phase.all_red_s, 3),
                }
            )

    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Calculate PKJI fixed-time plan")
    parser.add_argument("--input", required=True, help="PKJI intersection JSON input")
    parser.add_argument("--output", required=True, help="Output plan JSON")
    parser.add_argument(
        "--csv-output",
        default="",
        help="Optional phase summary CSV output",
    )
    args = parser.parse_args()

    intersections, corridors, settings = load_inputs(args.input)

    offsets = calculate_offsets_from_corridors(corridors)
    preferred_yellow_s = float(settings.get("preferred_yellow_s", 3.0))
    base_saturation = float(settings.get("base_saturation_flow_per_lane", 1800.0))
    clamp_cycle = bool(settings.get("clamp_cycle", True))

    plans = []
    for intersection in intersections:
        if intersection.intersection_id in offsets:
            intersection.offset_s = offsets[intersection.intersection_id]
        plan = calculate_intersection_plan(
            intersection,
            base_saturation_flow_per_lane=base_saturation,
            preferred_yellow_s=preferred_yellow_s,
            clamp_cycle=clamp_cycle,
        )
        plans.append(plan)

    output_payload = {
        "source_input": os.path.abspath(args.input),
        "settings": settings,
        "plans": [plan_to_dict(plan) for plan in plans],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_payload, f, indent=2)

    csv_output = args.csv_output
    if not csv_output:
        csv_output = str(output_path.with_suffix(".csv"))
    write_summary_csv(csv_output, plans)

    print(f"Wrote PKJI plan: {output_path}")
    print(f"Wrote phase summary: {csv_output}")
    for plan in plans:
        flags = []
        if plan.oversaturated:
            flags.append("oversaturated")
        if plan.cycle_clamped:
            flags.append("clamped")
        flag = f" ({', '.join(flags)})" if flags else ""
        print(
            f"{plan.intersection_id}: cycle={plan.cycle_s:.1f}s, "
            f"offset={plan.offset_s:.1f}s, "
            f"sum_y={plan.sum_critical_ratio:.3f}{flag}"
        )


if __name__ == "__main__":
    main()
