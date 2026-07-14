"""Create a SUMO delay_based grid network from the PKJI grid network.

This prepares the non-RL delay-based baseline for grid_3x3_dynamic. It does not
run evaluation; it only changes tlLogic metadata and green-phase duration bounds.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def is_green_phase(state: str) -> bool:
    lower = state.lower()
    return "g" in lower and "y" not in lower


def convert_tllogic(tl_logic: ET.Element, min_green: int, min_max_green: int) -> None:
    tl_logic.set("type", "delay_based")
    tl_logic.set("programID", "delay_based_program")

    for phase in tl_logic.findall("phase"):
        if not is_green_phase(phase.get("state", "")):
            continue
        duration = int(round(float(phase.get("duration", "1"))))
        phase.set("minDur", str(min(min_green, duration)))
        phase.set("maxDur", str(max(min_max_green, duration * 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create delay_based grid SUMO net")
    parser.add_argument(
        "--net-in",
        type=Path,
        default=PROJECT_ROOT / "data" / "networks" / "grid_3x3" / "grid_3x3_pkji_m1.net.xml",
    )
    parser.add_argument(
        "--net-out",
        type=Path,
        default=PROJECT_ROOT / "data" / "networks" / "grid_3x3" / "grid_3x3_delay_based.net.xml",
    )
    parser.add_argument("--min-green", type=int, default=8)
    parser.add_argument("--min-max-green", type=int, default=40)
    args = parser.parse_args()

    tree = ET.parse(args.net_in)
    root = tree.getroot()
    updated = []
    for tl_logic in root.findall("tlLogic"):
        convert_tllogic(tl_logic, args.min_green, args.min_max_green)
        updated.append(tl_logic.get("id", ""))

    args.net_out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.net_out, encoding="UTF-8", xml_declaration=True)

    print(f"Wrote delay_based grid net: {args.net_out}")
    print(f"Updated {len(updated)} traffic lights: {', '.join(updated)}")


if __name__ == "__main__":
    main()
