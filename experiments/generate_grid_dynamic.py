"""
Generate dynamic peak cross-traffic scenario for 3x3 Grid.
This stress-tests GAT-DQN vs IDQN on 2D topology with heavy cross-traffic and moving bottlenecks.

Features:
1. High turning ratios (forces intersection blocking).
2. Moving bottlenecks (3 distinct waves from different directions).
3. 9000 total vehicles (oversaturated).
"""

import argparse
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

VTYPES = {
    "passenger": {
        "vClass": "passenger", "length": "4.5", "minGap": "1.0",
        "accel": "2.6", "decel": "4.5", "sigma": "0.5", "maxSpeed": "13.89",
        "minGapLat": "0.6", "laneChangeModel": "SL2015", "color": "0,0,255", "prob": 0.35,
    },
    "motorcycle": {
        "vClass": "motorcycle", "length": "2.0", "minGap": "0.5",
        "accel": "3.5", "decel": "5.0", "sigma": "0.8", "maxSpeed": "11.11",
        "minGapLat": "0.2", "laneChangeModel": "SL2015", "lcSublane": "0.5", "color": "255,255,0", "prob": 0.65,
    },
}

# Define routes. Format: edge sequences.
# Center is B1.
# West: left0, left1, left2
# East: right0, right1, right2
# South: bottom0, bottom1, bottom2
# North: top0, top1, top2

WAVES = {
    # Wave 1: West and North attacking the grid (moving East and South), heavy turns at center
    "wave1": [
        "left1A1 A1B1 B1C1 C1right1",      # straight
        "left1A1 A1B1 B1B0 B0bottom1",    # turn right at center
        "left1A1 A1B1 B1B2 B2top1",       # turn left at center
        "top1B2 B2B1 B1B0 B0bottom1",     # straight
        "top1B2 B2B1 B1A1 A1left1",       # turn right at center
        "top1B2 B2B1 B1C1 C1right1",      # turn left at center
        "left0A0 A0B0 B0C0 C0right0",     # outer straight
        "top2C2 C2C1 C1C0 C0bottom2",     # outer straight
    ],
    # Wave 2: East and South attacking the grid (moving West and North)
    "wave2": [
        "right1C1 C1B1 B1A1 A1left1",     # straight
        "right1C1 C1B1 B1B2 B2top1",      # turn right at center
        "right1C1 C1B1 B1B0 B0bottom1",   # turn left at center
        "bottom1B0 B0B1 B1B2 B2top1",     # straight
        "bottom1B0 B0B1 B1C1 C1right1",   # turn right at center
        "bottom1B0 B0B1 B1A1 A1left1",    # turn left at center
        "right2C2 C2B2 B2A2 A2left2",     # outer straight
        "bottom0A0 A0A1 A1A2 A2top0",     # outer straight
    ],
    # Wave 3: Chaotic convergence from all 4 sides hitting the center and randomly leaving
    "wave3": [
        "left1A1 A1B1 B1C1 C1right1",
        "right1C1 C1B1 B1A1 A1left1",
        "top1B2 B2B1 B1B0 B0bottom1",
        "bottom1B0 B0B1 B1B2 B2top1",
        # Ringlock makers (turning across each other's path):
        "left1A1 A1A2 A2B2 B2B1 B1C1 C1right1",
        "right1C1 C1C0 C0B0 B0B1 B1A1 A1left1",
    ]
}

def generate(seed: int, output: str):
    rng = np.random.default_rng(seed)
    py_rng_seed = int(rng.integers(0, 2**31))
    import random
    py_rng = random.Random(py_rng_seed)

    root_el = ET.Element("routes")
    
    vtype_names = list(VTYPES.keys())
    vtype_probs = [VTYPES[k]["prob"] for k in vtype_names]
    for type_id, attrs in VTYPES.items():
        vtype_attrs = {k: v for k, v in attrs.items() if k != "prob"}
        vtype_attrs["id"] = type_id
        ET.SubElement(root_el, "vType", vtype_attrs)

    veh_id = 0
    
    def spawn_wave(routes, start_time, end_time, count):
        nonlocal veh_id
        depart_times = sorted(rng.uniform(start_time, end_time, count).tolist())
        for depart in depart_times:
            route = py_rng.choice(routes)
            vtype = py_rng.choices(vtype_names, weights=vtype_probs, k=1)[0]
            veh = ET.SubElement(root_el, "vehicle", {
                "id": f"dyn_{veh_id}",
                "depart": f"{depart:.2f}",
                "type": vtype,
                "departLane": "free",
            })
            ET.SubElement(veh, "route", {"edges": route})
            veh_id += 1

    # Total 5400 vehicles over 1 hour
    # Wave 1: 0 - 1200s (1800 vehs)
    # Wave 2: 1200 - 2400s (1800 vehs)
    # Wave 3: 2400 - 3600s (1800 vehs)
    spawn_wave(WAVES["wave1"], 0, 1200, 1800)
    spawn_wave(WAVES["wave2"], 1200, 2400, 1800)
    spawn_wave(WAVES["wave3"], 2400, 3600, 1800)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root_el).write(out_path, encoding="UTF-8", xml_declaration=True)
    
    print(f"Scenario : grid_3x3_dynamic (moving bottlenecks & cross-traffic)")
    print(f"Vehicles : {veh_id}")
    print(f"Seed     : {seed}")
    print(f"Output   : {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/networks/grid_3x3/grid_3x3_dynamic.rou.xml")
    args = parser.parse_args()
    generate(args.seed, args.output)
