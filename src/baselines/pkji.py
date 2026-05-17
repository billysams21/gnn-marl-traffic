"""
PKJI-inspired fixed-time traffic-signal calibration utilities.

The module keeps PKJI calculations separate from RL hyperparameters. It is meant
to build an engineering baseline that can be compared against learned policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


EMP_MP = 1.0
EMP_KS = 1.3
EMP_SM_PROTECTED = 0.15
EMP_SM_OPPOSED = 0.40
LTOR_MIN_WIDTH_M = 2.0


@dataclass
class Movement:
    movement_dir: str  # "L", "T", "R"
    vol_mp: float
    vol_sm: float
    vol_ks: float
    is_protected: bool
    ltor_allowed: bool
    ltor_lane_width_m: float
    phase_id: int
    saturation_flow_j: Optional[float] = None
    actual_green_s: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Movement":
        return cls(**data)


@dataclass
class Approach:
    name: str
    width_m: float
    num_lanes: int
    movements: List[Movement]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Approach":
        payload = dict(data)
        payload["movements"] = [
            Movement.from_dict(item) for item in payload.get("movements", [])
        ]
        return cls(**payload)


@dataclass
class Intersection:
    intersection_id: str
    approaches: List[Approach]
    num_phases: int
    offset_s: float = 0.0
    cycle_s: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Intersection":
        payload = dict(data)
        payload["approaches"] = [
            Approach.from_dict(item) for item in payload.get("approaches", [])
        ]
        return cls(**payload)


@dataclass
class CorridorLink:
    from_intersection: str
    to_intersection: str
    distance_m: float
    progression_speed_kmh: float
    main_direction: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorridorLink":
        return cls(**data)


@dataclass
class PhasePlan:
    phase_id: int
    critical_ratio: float
    green_s: float
    intergreen_s: float
    yellow_s: float
    all_red_s: float


@dataclass
class IntersectionPlan:
    intersection_id: str
    num_phases: int
    cycle_s: float
    offset_s: float
    total_lost_time_s: float
    sum_critical_ratio: float
    oversaturated: bool
    cycle_clamped: bool
    cycle_range_s: List[float]
    phases: List[PhasePlan] = field(default_factory=list)


def movement_smp(movement: Movement) -> float:
    """Convert movement volume from vehicles/hour to SMP/hour."""
    if (
        movement.movement_dir.upper() == "L"
        and movement.ltor_allowed
        and movement.ltor_lane_width_m >= LTOR_MIN_WIDTH_M
    ):
        return 0.0

    emp_sm = EMP_SM_PROTECTED if movement.is_protected else EMP_SM_OPPOSED
    return (
        movement.vol_mp * EMP_MP
        + movement.vol_sm * emp_sm
        + movement.vol_ks * EMP_KS
    )


def default_saturation_flow(approach: Approach, base_per_lane: float) -> float:
    """Fallback saturation flow when measured/calculated J is unavailable."""
    return base_per_lane * max(approach.num_lanes, 1)


def normal_intergreen_s(width_m: float) -> float:
    """PKJI normal intergreen bucket from average approach width."""
    if width_m < 10.0:
        return 4.0
    if width_m < 15.0:
        return 5.0
    return 6.0


def feasible_cycle_range(num_phases: int) -> List[float]:
    if num_phases <= 2:
        return [40.0, 80.0]
    if num_phases == 3:
        return [50.0, 100.0]
    return [80.0, 130.0]


def yellow_all_red_split(intergreen_s: float, preferred_yellow_s: float) -> tuple:
    yellow_s = min(preferred_yellow_s, intergreen_s)
    all_red_s = max(intergreen_s - yellow_s, 0.0)
    return yellow_s, all_red_s


def calculate_intersection_plan(
    intersection: Intersection,
    base_saturation_flow_per_lane: float = 1800.0,
    preferred_yellow_s: float = 3.0,
    clamp_cycle: bool = True,
) -> IntersectionPlan:
    phase_ratios = {phase_id: 0.0 for phase_id in range(1, intersection.num_phases + 1)}
    phase_intergreens = {
        phase_id: 0.0 for phase_id in range(1, intersection.num_phases + 1)
    }

    for approach in intersection.approaches:
        approach_intergreen = normal_intergreen_s(approach.width_m)
        for movement in approach.movements:
            phase_id = movement.phase_id
            if phase_id not in phase_ratios:
                raise ValueError(
                    f"{intersection.intersection_id}: movement phase_id={phase_id} "
                    f"outside 1..{intersection.num_phases}"
                )

            smp = movement_smp(movement)
            saturation = movement.saturation_flow_j or default_saturation_flow(
                approach, base_saturation_flow_per_lane
            )
            if saturation <= 0:
                raise ValueError(
                    f"{intersection.intersection_id}: saturation_flow_j must be > 0"
                )

            phase_ratios[phase_id] = max(phase_ratios[phase_id], smp / saturation)
            phase_intergreens[phase_id] = max(
                phase_intergreens[phase_id], approach_intergreen
            )

    total_lost_time = sum(phase_intergreens.values())
    sum_ratio = sum(phase_ratios.values())
    denominator = 1.0 - sum_ratio
    oversaturated = denominator <= 0.0

    if intersection.cycle_s is not None:
        cycle_s = float(intersection.cycle_s)
    elif oversaturated:
        cycle_s = feasible_cycle_range(intersection.num_phases)[1]
    else:
        cycle_s = (1.5 * total_lost_time + 5.0) / denominator

    cycle_range = feasible_cycle_range(intersection.num_phases)
    cycle_clamped = False
    if cycle_s < cycle_range[0] or cycle_s > cycle_range[1]:
        if cycle_s > cycle_range[1]:
            oversaturated = True
        if clamp_cycle:
            cycle_clamped = True
            cycle_s = min(max(cycle_s, cycle_range[0]), cycle_range[1])

    available_green = max(cycle_s - total_lost_time, 0.0)
    phases: List[PhasePlan] = []
    for phase_id in range(1, intersection.num_phases + 1):
        ratio = phase_ratios[phase_id]
        if sum_ratio > 0:
            green_s = available_green * (ratio / sum_ratio)
        else:
            green_s = available_green / max(intersection.num_phases, 1)

        intergreen_s = phase_intergreens[phase_id]
        yellow_s, all_red_s = yellow_all_red_split(intergreen_s, preferred_yellow_s)
        phases.append(
            PhasePlan(
                phase_id=phase_id,
                critical_ratio=ratio,
                green_s=green_s,
                intergreen_s=intergreen_s,
                yellow_s=yellow_s,
                all_red_s=all_red_s,
            )
        )

    return IntersectionPlan(
        intersection_id=intersection.intersection_id,
        num_phases=intersection.num_phases,
        cycle_s=cycle_s,
        offset_s=intersection.offset_s,
        total_lost_time_s=total_lost_time,
        sum_critical_ratio=sum_ratio,
        oversaturated=oversaturated,
        cycle_clamped=cycle_clamped,
        cycle_range_s=cycle_range,
        phases=phases,
    )


def calculate_offsets_from_corridors(
    links: List[CorridorLink],
    origin_offsets: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute simple progression offsets from corridor distance/speed."""
    offsets = dict(origin_offsets or {})
    for link in links:
        if link.progression_speed_kmh <= 0:
            raise ValueError("progression_speed_kmh must be > 0")
        from_offset = offsets.get(link.from_intersection, 0.0)
        speed_mps = link.progression_speed_kmh / 3.6
        travel_time_s = link.distance_m / speed_mps
        offsets[link.from_intersection] = from_offset
        offsets[link.to_intersection] = from_offset + travel_time_s
    return offsets


def plan_to_dict(plan: IntersectionPlan) -> Dict[str, Any]:
    return {
        "intersection_id": plan.intersection_id,
        "num_phases": plan.num_phases,
        "cycle_s": round(plan.cycle_s, 3),
        "offset_s": round(plan.offset_s, 3),
        "total_lost_time_s": round(plan.total_lost_time_s, 3),
        "sum_critical_ratio": round(plan.sum_critical_ratio, 6),
        "oversaturated": plan.oversaturated,
        "cycle_clamped": plan.cycle_clamped,
        "cycle_range_s": plan.cycle_range_s,
        "phases": [
            {
                "phase_id": phase.phase_id,
                "critical_ratio": round(phase.critical_ratio, 6),
                "green_s": round(phase.green_s, 3),
                "intergreen_s": round(phase.intergreen_s, 3),
                "yellow_s": round(phase.yellow_s, 3),
                "all_red_s": round(phase.all_red_s, 3),
            }
            for phase in plan.phases
        ],
    }
