"""
Generate SUMO route file for the arterial+collector network — directional imbalance scenario.

Scenario: arterial_unbalanced
  - Total vehicles: 6000 (same as stable, isolate demand distribution effect)
  - Spawn: uniform over 3600s (same as stable)
  - Key difference: E->W arteri demand = 3x W->E arteri demand
    (simulates morning commute into city center on west side)

Route weight changes vs stable:
  - W_A0->E_A4 (W->E): weight 35 -> 10  (reduced)
  - E_A4->W_A0 (E->W): weight 35 -> 60  (dominant direction)
  - Feeder/cross/collector weights kept proportional but scaled down
    to preserve total weight budget = 100

Vehicle types: same as stable (35% passenger, 65% motorcycle, PKJI-compliant)
"""

import argparse
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path


# ---------------------------------------------------------------------------
# Vehicle type definitions (identical to generate_arterial_routes.py)
# ---------------------------------------------------------------------------
VTYPES = {
    "passenger": {
        "vClass":        "passenger",
        "length":        "4.5",
        "minGap":        "1.0",
        "accel":         "2.6",
        "decel":         "4.5",
        "sigma":         "0.5",
        "maxSpeed":      "13.89",
        "minGapLat":     "0.6",
        "laneChangeModel": "SL2015",
        "color":         "0,0,255",
        "prob":          0.35,
    },
    "motorcycle": {
        "vClass":        "motorcycle",
        "length":        "2.0",
        "minGap":        "0.5",
        "accel":         "3.5",
        "decel":         "5.0",
        "sigma":         "0.8",
        "maxSpeed":      "11.11",
        "minGapLat":     "0.2",
        "laneChangeModel": "SL2015",
        "lcSublane":     "0.5",
        "color":         "255,255,0",
        "prob":          0.65,
    },
}

# ---------------------------------------------------------------------------
# Route templates: (edge_sequence, weight)
# Unbalanced: E->W dominates (weight 60), W->E minor (weight 10)
# Feeder routes adjusted: more feeders exit west (W_A0), fewer exit east (E_A4)
# Total weights sum to 100 for easy interpretation as percentages
# ---------------------------------------------------------------------------
ROUTE_TEMPLATES_UNBALANCED = [
    # 1. Arteri Utama — E->W dominates (morning commute into western CBD)
    ("W_A0A1 A1A2 A2A3 A3E_A4",  10),   # W->E: minor (was 35)
    ("E_A4A3 A3A2 A2A1 A1W_A0",  60),   # E->W: dominant (was 35)

    # 2. Kolektor Horizontal — kept symmetric, scaled down
    ("W_C1C1 C1C2 C2C3 C3E_C3",   2),
    ("E_C3C3 C3C2 C2C1 C1W_C1",   2),
    ("W_B1B1 B1B2 B2B3 B3E_B3",   2),
    ("E_B3B3 B3B2 B2B1 B1W_B1",   2),

    # 3. North/South feeder -> Arteri ke Barat (dominant direction)
    ("N1C1 C1A1 A1W_A0",                    3),
    ("N2C2 C2A2 A2A1 A1W_A0",               3),
    ("N3C3 C3A3 A3A2 A2A1 A1W_A0",          2),
    ("S1B1 B1A1 A1W_A0",                    3),
    ("S2B2 B2A2 A2A1 A1W_A0",               3),
    ("S3B3 B3A3 A3A2 A2A1 A1W_A0",          2),

    # 4. North/South feeder -> Arteri ke Timur (minor)
    ("N1C1 C1A1 A1A2 A2A3 A3E_A4",  1),
    ("S1B1 B1A1 A1A2 A2A3 A3E_A4",  1),

    # 5. Arteri Timur -> keluar ke North/South
    ("E_A4A3 A3C3 C3N3",                    1),
    ("E_A4A3 A3A2 A2C2 C2N2",               1),
    ("E_A4A3 A3B3 B3S3",                    1),
    ("E_A4A3 A3A2 A2B2 B2S2",               1),

    # 6. Cross Traffic (North <-> South) — unchanged
    ("N1C1 C1A1 A1B1 B1S1",  1),
    ("S1B1 B1A1 A1C1 C1N1",  1),
    ("N2C2 C2A2 A2B2 B2S2",  1),
    ("S2B2 B2A2 A2C2 C2N2",  1),
    ("N3C3 C3A3 A3B3 B3S3",  1),
    ("S3B3 B3A3 A3C3 C3N3",  1),
]


def generate(seed: int, output: str):
    rng = np.random.default_rng(seed)
    py_rng_seed = int(rng.integers(0, 2**31))
    import random
    py_rng = random.Random(py_rng_seed)

    total_vehicles = 6000
    depart_times = sorted(rng.uniform(0, 3600, total_vehicles).tolist())

    routes, weights = zip(*ROUTE_TEMPLATES_UNBALANCED)
    total_weight = sum(weights)
    probs = [w / total_weight for w in weights]

    vtype_names = list(VTYPES.keys())
    vtype_probs = [VTYPES[k]["prob"] for k in vtype_names]

    root_el = ET.Element("routes")

    for type_id, attrs in VTYPES.items():
        vtype_attrs = {k: v for k, v in attrs.items() if k != "prob"}
        vtype_attrs["id"] = type_id
        ET.SubElement(root_el, "vType", vtype_attrs)

    for i, depart in enumerate(depart_times):
        route = py_rng.choices(routes, weights=probs, k=1)[0]
        vtype = py_rng.choices(vtype_names, weights=vtype_probs, k=1)[0]
        veh = ET.SubElement(root_el, "vehicle", {
            "id": str(i),
            "depart": f"{depart:.2f}",
            "type": vtype,
            "departLane": "free",
        })
        ET.SubElement(veh, "route", {"edges": route})

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root_el).write(out_path, encoding="UTF-8", xml_declaration=True)

    # Print demand summary
    ew = sum(w for r, w in ROUTE_TEMPLATES_UNBALANCED if r.startswith("W_A0"))
    we = sum(w for r, w in ROUTE_TEMPLATES_UNBALANCED if r.startswith("E_A4"))
    print(f"Scenario : arterial_unbalanced (directional imbalance)")
    print(f"Vehicles : {total_vehicles}")
    print(f"Seed     : {seed}")
    print(f"W->E arteri weight: {ew}/{total_weight} = {ew/total_weight*100:.1f}%")
    print(f"E->W arteri weight: {we}/{total_weight} = {we/total_weight*100:.1f}%")
    print(f"E->W / W->E ratio : {we/ew:.1f}x")
    print(f"Output   : {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate arterial unbalanced route file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="data/networks/arterial_3x3/arterial_unbalanced.rou.xml",
    )
    args = parser.parse_args()
    generate(args.seed, args.output)
