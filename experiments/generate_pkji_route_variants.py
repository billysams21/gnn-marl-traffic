"""
Generate PKJI-aware SUMO route variants from an existing route file.

The script preserves the original OD/path distribution, then:
  - assigns vehicle classes (passenger/motorcycle/heavy) by probability,
  - optionally scales demand by duplicating vehicles with small depart jitter,
  - writes SUMO vType definitions for different vehicle behavior/geometry.

This is intended for apple-to-apple RL vs fixed-time comparisons on the same
PKJI-aware demand scenario.
"""

import argparse
import copy
import random
import xml.etree.ElementTree as ET
from pathlib import Path


VTYPE_DEFS = {
    "passenger": {
        "vClass": "passenger",
        "length": "4.5",
        "minGap": "2.5",
        "accel": "2.6",
        "decel": "4.5",
        "sigma": "0.5",
        "maxSpeed": "13.89",
        "color": "0,0,255",
    },
    "motorcycle": {
        "vClass": "motorcycle",
        "length": "2.0",
        "minGap": "1.0",
        "accel": "3.5",
        "decel": "5.0",
        "sigma": "0.7",
        "maxSpeed": "16.67",
        "color": "255,255,0",
    },
    "heavy": {
        "vClass": "truck",
        "length": "8.0",
        "minGap": "3.0",
        "accel": "1.2",
        "decel": "4.0",
        "sigma": "0.4",
        "maxSpeed": "11.11",
        "color": "255,0,0",
    },
}


def pick_vehicle_type(rng: random.Random, p_motorcycle: float, p_heavy: float) -> str:
    x = rng.random()
    if x < p_motorcycle:
        return "motorcycle"
    if x < p_motorcycle + p_heavy:
        return "heavy"
    return "passenger"


def vehicle_sort_key(vehicle: ET.Element):
    return float(vehicle.get("depart", "0")), vehicle.get("id", "")


def clone_vehicle(vehicle: ET.Element, new_id: str, depart: float, type_id: str):
    clone = copy.deepcopy(vehicle)
    clone.set("id", new_id)
    clone.set("depart", f"{depart:.2f}")
    clone.set("type", type_id)
    return clone


def main():
    parser = argparse.ArgumentParser(description="Generate PKJI-aware route variant")
    parser.add_argument("--route-in", required=True)
    parser.add_argument("--route-out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demand-multiplier", type=float, default=1.0)
    parser.add_argument("--motorcycle-share", type=float, default=0.65)
    parser.add_argument("--heavy-share", type=float, default=0.05)
    parser.add_argument("--depart-jitter-s", type=float, default=0.8)
    args = parser.parse_args()

    if not (0 <= args.motorcycle_share <= 1):
        raise ValueError("--motorcycle-share must be in [0, 1]")
    if not (0 <= args.heavy_share <= 1):
        raise ValueError("--heavy-share must be in [0, 1]")
    if args.motorcycle_share + args.heavy_share > 1:
        raise ValueError("motorcycle-share + heavy-share must be <= 1")
    if args.demand_multiplier <= 0:
        raise ValueError("--demand-multiplier must be > 0")

    rng = random.Random(args.seed)
    tree = ET.parse(args.route_in)
    root = tree.getroot()

    original_vehicles = list(root.findall("vehicle"))
    for child in list(root):
        if child.tag in ("vehicle", "vType"):
            root.remove(child)

    for type_id, attrs in VTYPE_DEFS.items():
        ET.SubElement(root, "vType", {"id": type_id, **attrs})

    base_copies = int(args.demand_multiplier)
    fractional = args.demand_multiplier - base_copies
    generated = []

    for vehicle in original_vehicles:
        original_depart = float(vehicle.get("depart", "0"))
        copies = base_copies
        if rng.random() < fractional:
            copies += 1

        for copy_idx in range(copies):
            type_id = pick_vehicle_type(
                rng, args.motorcycle_share, args.heavy_share
            )
            jitter = 0.0 if copy_idx == 0 else rng.uniform(0, args.depart_jitter_s)
            depart = original_depart + copy_idx * args.depart_jitter_s + jitter
            new_id = (
                vehicle.get("id", "veh")
                if copy_idx == 0
                else f"{vehicle.get('id', 'veh')}_x{copy_idx}"
            )
            generated.append(clone_vehicle(vehicle, new_id, depart, type_id))

    for vehicle in sorted(generated, key=vehicle_sort_key):
        root.append(vehicle)

    out_path = Path(args.route_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="UTF-8", xml_declaration=True)

    counts = {key: 0 for key in VTYPE_DEFS}
    for vehicle in generated:
        counts[vehicle.get("type")] += 1

    print(f"Wrote route variant: {out_path}")
    print(f"Original vehicles: {len(original_vehicles)}")
    print(f"Generated vehicles: {len(generated)}")
    print(f"Vehicle type counts: {counts}")


if __name__ == "__main__":
    main()
