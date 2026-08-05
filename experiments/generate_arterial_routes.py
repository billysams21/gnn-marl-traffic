"""
Generate SUMO route file for the arterial+collector network.

Two scenarios:
  - stable:  6000 vehicles, uniform spawn over 3600s (Poisson-like flat demand)
    - peak:    9000 vehicles, Gaussian spawn centered at 1800s (rush-hour wave)

Vehicle types (no trucks, PKJI-compliant EMP):
  - passenger : 35%  EMP=1.0
  - motorcycle: 65%  EMP=0.2

Sublane model parameters included for realistic motorcycle lane-splitting.
"""

import argparse
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path


# ---------------------------------------------------------------------------
# Vehicle type definitions
# ---------------------------------------------------------------------------
VTYPES = {
    "passenger": {
        "vClass":        "passenger",
        "length":        "4.5",
        "minGap":        "1.0",
        "accel":         "2.6",
        "decel":         "4.5",
        "sigma":         "0.5",
        "maxSpeed":      "13.89",   # ~50 km/h
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
        "sigma":         "0.8",     # more random/aggressive
        "maxSpeed":      "11.11",   # ~40 km/h (realistic urban Indonesia)
        "minGapLat":     "0.2",     # lane-splitting gap
        "laneChangeModel": "SL2015",
        "lcSublane":     "0.5",     # willingness to use sublane
        "color":         "255,255,0",
        "prob":          0.65,
    },
}

# ---------------------------------------------------------------------------
# Route templates: (edge_sequence, relative_weight)
# All edges verified against arterial.net.xml
# ---------------------------------------------------------------------------
ROUTE_TEMPLATES = [
    # 1. Arteri Utama (dominant)
    ("W_A0A1 A1A2 A2A3 A3E_A4",  35),
    ("E_A4A3 A3A2 A2A1 A1W_A0",  35),

    # 2. Kolektor Horizontal (minor)
    ("W_C1C1 C1C2 C2C3 C3E_C3",   5),
    ("E_C3C3 C3C2 C2C1 C1W_C1",   5),
    ("W_B1B1 B1B2 B2B3 B3E_B3",   5),
    ("E_B3B3 B3B2 B2B1 B1W_B1",   5),

    # 3. North/South feeder -> Arteri ke Timur
    ("N1C1 C1A1 A1A2 A2A3 A3E_A4", 3),
    ("N2C2 C2A2 A2A3 A3E_A4",      3),
    ("N3C3 C3A3 A3E_A4",            2),
    ("S1B1 B1A1 A1A2 A2A3 A3E_A4", 3),
    ("S2B2 B2A2 A2A3 A3E_A4",      3),
    ("S3B3 B3A3 A3E_A4",            2),

    # 4. Arteri Barat -> keluar ke North/South
    ("W_A0A1 A1C1 C1N1",                    3),
    ("W_A0A1 A1A2 A2C2 C2N2",               3),
    ("W_A0A1 A1A2 A2A3 A3C3 C3N3",          2),
    ("W_A0A1 A1B1 B1S1",                    3),
    ("W_A0A1 A1A2 A2B2 B2S2",               3),
    ("W_A0A1 A1A2 A2A3 A3B3 B3S3",          2),

    # 5. Cross Traffic (North <-> South)
    ("N1C1 C1A1 A1B1 B1S1",  1),
    ("S1B1 B1A1 A1C1 C1N1",  1),
    ("N2C2 C2A2 A2B2 B2S2",  1),
    ("S2B2 B2A2 A2C2 C2N2",  1),
    ("N3C3 C3A3 A3B3 B3S3",  1),
    ("S3B3 B3A3 A3C3 C3N3",  1),
]


def generate(scenario: str, seed: int, output: str):
    rng = np.random.default_rng(seed)
    py_rng_seed = int(rng.integers(0, 2**31))
    import random
    py_rng = random.Random(py_rng_seed)

    if scenario == "stable":
        total_vehicles = 6000
        depart_times = sorted(rng.uniform(0, 3600, total_vehicles).tolist())
    elif scenario == "peak":
        total_vehicles = 9000
        raw = rng.normal(loc=1800, scale=700, size=total_vehicles)
        depart_times = sorted(np.clip(raw, 0, 3599).tolist())
    else:
        raise ValueError(f"Unknown scenario: {scenario}. Use 'stable' or 'peak'.")

    routes, weights = zip(*ROUTE_TEMPLATES)
    total_weight = sum(weights)
    probs = [w / total_weight for w in weights]

    vtype_names = list(VTYPES.keys())
    vtype_probs = [VTYPES[k]["prob"] for k in vtype_names]

    root_el = ET.Element("routes")

    # Write vType definitions
    for type_id, attrs in VTYPES.items():
        vtype_attrs = {k: v for k, v in attrs.items() if k != "prob"}
        vtype_attrs["id"] = type_id
        ET.SubElement(root_el, "vType", vtype_attrs)

    # Write vehicles
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
    print(f"Scenario : {scenario}")
    print(f"Vehicles : {total_vehicles}")
    print(f"Seed     : {seed}")
    print(f"Output   : {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate arterial route file")
    parser.add_argument("--scenario", choices=["stable", "peak"], default="stable")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/networks/arterial_3x3/arterial_stable.rou.xml")
    args = parser.parse_args()
    generate(args.scenario, args.seed, args.output)
