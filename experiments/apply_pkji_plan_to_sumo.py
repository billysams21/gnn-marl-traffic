"""
Apply a calculated PKJI fixed-time plan to a SUMO .net.xml file.

The script preserves the existing green/yellow phase states and updates:
    - tlLogic offset
    - green durations per phase
    - yellow durations
    - optional all-red phases after yellow phases
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def is_green_phase(state: str) -> bool:
    lower = state.lower()
    return ("g" in lower) and ("y" not in lower)


def is_yellow_phase(state: str) -> bool:
    return "y" in state.lower()


def all_red_state(length: int) -> str:
    return "r" * length


def load_plan(path: str):
    with open(path, "r") as f:
        payload = json.load(f)
    return {item["intersection_id"]: item for item in payload.get("plans", [])}


def phase_duration(value: float, minimum: int = 1) -> str:
    return str(max(int(round(value)), minimum))


def replace_tllogic(tl_logic: ET.Element, plan: dict, include_all_red: bool):
    tl_logic.set("offset", phase_duration(plan.get("offset_s", 0), minimum=0))

    existing_phases = list(tl_logic.findall("phase"))
    green_phases = [phase for phase in existing_phases if is_green_phase(phase.get("state", ""))]
    yellow_phases = [phase for phase in existing_phases if is_yellow_phase(phase.get("state", ""))]
    plan_phases = sorted(plan.get("phases", []), key=lambda item: item["phase_id"])

    if len(green_phases) < len(plan_phases):
        raise ValueError(
            f"{tl_logic.get('id')}: net has {len(green_phases)} green phases, "
            f"but plan has {len(plan_phases)} phases"
        )

    for child in list(tl_logic):
        if child.tag == "phase":
            tl_logic.remove(child)

    for idx, phase_plan in enumerate(plan_phases):
        green_state = green_phases[idx].get("state")
        yellow_state = (
            yellow_phases[idx].get("state")
            if idx < len(yellow_phases)
            else green_state.replace("G", "y").replace("g", "y")
        )

        ET.SubElement(
            tl_logic,
            "phase",
            {
                "duration": phase_duration(phase_plan["green_s"]),
                "state": green_state,
            },
        )
        ET.SubElement(
            tl_logic,
            "phase",
            {
                "duration": phase_duration(phase_plan["yellow_s"]),
                "state": yellow_state,
            },
        )
        all_red_s = float(phase_plan.get("all_red_s", 0.0))
        if include_all_red and all_red_s > 0:
            ET.SubElement(
                tl_logic,
                "phase",
                {
                    "duration": phase_duration(all_red_s),
                    "state": all_red_state(len(green_state)),
                },
            )


def main():
    parser = argparse.ArgumentParser(description="Apply PKJI plan to SUMO net")
    parser.add_argument("--net-in", required=True, help="Input .net.xml")
    parser.add_argument("--plan", required=True, help="Plan JSON from calculate script")
    parser.add_argument("--net-out", required=True, help="Output calibrated .net.xml")
    parser.add_argument(
        "--no-all-red",
        action="store_true",
        help="Do not insert all-red phases even when plan contains all_red_s",
    )
    args = parser.parse_args()

    plans = load_plan(args.plan)
    tree = ET.parse(args.net_in)
    root = tree.getroot()

    updated = []
    for tl_logic in root.findall("tlLogic"):
        tl_id = tl_logic.get("id")
        if tl_id not in plans:
            continue
        replace_tllogic(tl_logic, plans[tl_id], include_all_red=not args.no_all_red)
        updated.append(tl_id)

    if not updated:
        raise RuntimeError("No tlLogic elements matched plan intersections.")

    out_path = Path(args.net_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="UTF-8", xml_declaration=True)

    print(f"Wrote calibrated SUMO net: {out_path}")
    print(f"Updated {len(updated)} traffic lights: {', '.join(updated)}")


if __name__ == "__main__":
    main()
