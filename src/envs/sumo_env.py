"""
SUMO Multi-Agent Traffic Signal Control Environment.
Wraps SUMO simulator via TraCI for GNN-MARL training.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Union

# SUMO setup
if "SUMO_HOME" in os.environ:
    os.environ["SUMO_HOME"] = os.environ["SUMO_HOME"].strip()
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    # Default SUMO path on Windows
    sumo_home = r"C:\Program Files (x86)\Sumo"
    os.environ["SUMO_HOME"] = sumo_home
    sys.path.append(os.path.join(sumo_home, "tools"))

import traci
import sumolib


def _normalize_path(path: Optional[str]) -> Optional[str]:
    """Normalize filesystem paths before passing them to SUMO/sumolib."""
    return os.path.normpath(path) if path else None


def _ensure_file(path: Optional[str], label: str):
    """Fail early with a readable error for missing scenario files."""
    if path and not os.path.isfile(path):
        raise FileNotFoundError(
            f"{label} not found: {path}. "
            "If this is a TorontoSUMONetworks scenario, export/copy the generated "
            "files into data/networks/toronto_small first."
        )


class SumoEnvironment:
    """
    Multi-agent environment for traffic signal control using SUMO.

    Each traffic light is an agent. The environment provides:
    - State: queue lengths, densities, waiting times, phase info, and temporal deltas
    - Actions: select a traffic light phase
    - Reward: negative weighted sum of queue lengths and waiting times

    === OBSERVATION NORMALIZATION (Best Practice) ===
    All features are normalized for stable neural network training:
    
    - queue_norm:        queue / max_queue_per_lane     → [0, 1]
    - delta_queue_norm:  delta_queue / max_delta_queue  → [-1, 1]
    - density_norm:      density (already 0-1)          → [0, 1]
    - waiting_norm:      waiting / max_waiting_time     → [0, 1]
    - delta_density_norm: delta / max_delta_density     → [-1, 1]
    - phase_onehot:      one-hot encoded phase          → {0, 1}
    - phase_duration:    duration / max_green           → [0, 1]

    This follows standard deep learning practice: all inputs scaled to similar ranges
    prevents features with large magnitudes (waiting_time ~300s) from dominating
    gradients over features with small magnitudes (queue ~0-30).
    """

    def __init__(
        self,
        net_file: str,
        route_file: Optional[Union[str, List[str]]] = None,
        sumocfg_file: Optional[str] = None,
        num_seconds: int = 3600,
        delta_time: int = 5,
        yellow_time: int = 2,
        min_green: int = 10,
        max_green: int = 60,
        time_to_teleport: int = -1,
        recovery_enabled: bool = False,
        recovery_enter_delay: float = 30.0,
        recovery_enter_queue: float = 3.0,
        recovery_exit_delay: float = 10.0,
        recovery_exit_queue: float = 1.5,
        recovery_hold_seconds: int = 30,
        reward_alpha: float = 0.5,
        use_gui: bool = False,
        seed: int = 42,
        # Normalization parameters (best practice from GAP_AUDIT)
        max_queue_per_lane: int = 30,      # Max vehicles that can queue per lane
        max_waiting_time: float = 300.0,   # Max waiting time in seconds (5 min)
        max_delta_queue: float = 10.0,     # Max expected queue change per step
        max_delta_density: float = 0.5,    # Max expected density change per step
    ):
        self.net_file = _normalize_path(net_file)
        if isinstance(route_file, list):
            self.route_file = [_normalize_path(path) for path in route_file]
        else:
            self.route_file = _normalize_path(route_file)
        self.sumocfg_file = _normalize_path(sumocfg_file)
        self.num_seconds = num_seconds
        self.delta_time = delta_time  # seconds between agent decisions
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.time_to_teleport = time_to_teleport
        self.recovery_enabled = recovery_enabled
        self.recovery_enter_delay = recovery_enter_delay
        self.recovery_enter_queue = recovery_enter_queue
        self.recovery_exit_delay = recovery_exit_delay
        self.recovery_exit_queue = recovery_exit_queue
        self.recovery_hold_steps = max(
            1, int(np.ceil(recovery_hold_seconds / max(self.delta_time, 1)))
        )
        self.reward_alpha = reward_alpha
        self.use_gui = use_gui
        self.seed = seed

        # Normalization constants
        self.max_queue_per_lane = max_queue_per_lane
        self.max_waiting_time = max_waiting_time
        self.max_delta_queue = max_delta_queue
        self.max_delta_density = max_delta_density

        # Per-agent info
        self.controlled_lanes: Dict[str, List[str]] = {}
        self.num_phases: Dict[str, int] = {}
        self.phase_defs: Dict[str, List[str]] = {}
        self._phase_lanes: Dict[str, List[List[str]]] = {}

        # State tracking for temporal features
        self._prev_queue: Dict[str, np.ndarray] = {}
        self._prev_density: Dict[str, np.ndarray] = {}

        # Timing
        self._step_count = 0
        self._yellow_phase_active: Dict[str, bool] = {}
        self._current_phase: Dict[str, int] = {}
        self._phase_duration: Dict[str, int] = {}
        self._pending_phase: Dict[str, Optional[int]] = {}
        self._yellow_remaining: Dict[str, int] = {}
        self._episode_arrived: int = 0
        self._episode_departed: int = 0
        self._episode_emergency_stops: int = 0
        self._recovery_active: bool = False
        self._recovery_enter_counter: int = 0
        self._recovery_exit_counter: int = 0

        self._sumo_running = False

        if not self.sumocfg_file and not self.route_file:
            raise ValueError("Either route_file or sumocfg_file must be provided.")

        _ensure_file(self.net_file, "SUMO net file")
        _ensure_file(self.sumocfg_file, "SUMO config file")
        if isinstance(self.route_file, list):
            for path in self.route_file:
                _ensure_file(path, "SUMO route file")
        else:
            _ensure_file(self.route_file, "SUMO route file")

        # Load network to get topology info
        self.net = sumolib.net.readNet(self.net_file)

        # Get traffic light IDs
        self.ts_ids: List[str] = [tl.getID() for tl in self.net.getTrafficLights()]
        self.num_agents = len(self.ts_ids)

        # Build adjacency info from network topology
        self.adjacency_matrix = self._build_adjacency()

    def _build_adjacency(self) -> np.ndarray:
        """Build adjacency matrix from network topology (shared edges between TL nodes)."""
        n = self.num_agents
        adj = np.eye(n, dtype=np.float32)  # self-loops

        tl_nodes = {}
        for i, tl_id in enumerate(self.ts_ids):
            tl = self.net.getTrafficLights()
            for t in tl:
                if t.getID() == tl_id:
                    # Get the node(s) controlled by this TL
                    connections = t.getConnections()
                    node_ids = set()
                    for conn in connections:
                        from_lane = conn[0]
                        from_edge = from_lane.getEdge()
                        to_lane = conn[1]
                        to_edge = to_lane.getEdge()
                        node_ids.add(from_edge.getToNode().getID())
                        node_ids.add(to_edge.getFromNode().getID())
                    tl_nodes[i] = node_ids
                    break

        # Two TLs are adjacent if they share a connecting edge
        for i in range(n):
            for j in range(i + 1, n):
                nodes_i = tl_nodes.get(i, set())
                nodes_j = tl_nodes.get(j, set())

                # Check if any node from i connects to any node from j
                connected = False
                for ni in nodes_i:
                    node = self.net.getNode(ni)
                    for edge in node.getOutgoing():
                        if edge.getToNode().getID() in nodes_j:
                            connected = True
                            break
                    if not connected:
                        for edge in node.getIncoming():
                            if edge.getFromNode().getID() in nodes_j:
                                connected = True
                                break
                    if connected:
                        break

                if connected:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

        return adj

    def _get_sumo_cmd(self) -> List[str]:
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_path = os.path.join(os.environ["SUMO_HOME"], "bin", sumo_binary)
        if sys.platform == "win32":
            sumo_path += ".exe"

        if self.sumocfg_file:
            cmd = [sumo_path, "-c", self.sumocfg_file]
        else:
            route_files = self.route_file
            if isinstance(route_files, list):
                route_files = ",".join(route_files)
            cmd = [sumo_path, "-n", self.net_file, "-r", str(route_files)]

        cmd.extend(
            [
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", str(self.time_to_teleport),
            "--seed", str(self.seed),
            ]
        )
        return cmd

    def reset(self, preserve_default_program: bool = False) -> Dict[str, np.ndarray]:
        """Reset environment and return initial observations."""
        if self._sumo_running:
            traci.close()

        traci.start(self._get_sumo_cmd())
        self._sumo_running = True
        self._step_count = 0
        self._episode_arrived = 0
        self._episode_departed = 0
        self._episode_emergency_stops = 0
        self._recovery_active = False
        self._recovery_enter_counter = 0
        self._recovery_exit_counter = 0

        # Initialize per-TL info from running simulation
        for ts_id in self.ts_ids:
            lanes = traci.trafficlight.getControlledLanes(ts_id)
            # Deterministic lane ordering across runs (no set-based nondeterminism).
            self.controlled_lanes[ts_id] = sorted(dict.fromkeys(lanes))
            logic = traci.trafficlight.getAllProgramLogics(ts_id)[0]
            phases = logic.getPhases()
            # Filter only green phases (non-yellow, non-all-red)
            green_phases = []
            for p in phases:
                state = p.state
                if "G" in state or "g" in state:
                    if "y" not in state.lower():
                        green_phases.append(state)
            self.phase_defs[ts_id] = green_phases if green_phases else [phases[0].state]
            self.num_phases[ts_id] = len(self.phase_defs[ts_id])
            self._phase_lanes[ts_id] = self._build_phase_lane_map(ts_id)

            self._current_phase[ts_id] = 0
            self._phase_duration[ts_id] = 0
            self._yellow_phase_active[ts_id] = False
            self._pending_phase[ts_id] = None
            self._yellow_remaining[ts_id] = 0

            if preserve_default_program:
                current_state = traci.trafficlight.getRedYellowGreenState(ts_id)
                for phase_idx, phase_state in enumerate(self.phase_defs[ts_id]):
                    if phase_state == current_state:
                        self._current_phase[ts_id] = phase_idx
                        break
            else:
                # Ensure each signal starts with a deterministic initial green phase.
                traci.trafficlight.setRedYellowGreenState(ts_id, self.phase_defs[ts_id][0])

        # Initialize previous state for delta computation
        obs = {}
        for ts_id in self.ts_ids:
            lanes = self.controlled_lanes[ts_id]
            self._prev_queue[ts_id] = np.zeros(len(lanes), dtype=np.float32)
            self._prev_density[ts_id] = np.zeros(len(lanes), dtype=np.float32)

        # Run a few warmup steps
        for _ in range(self.delta_time):
            traci.simulationStep()
            self._step_count += 1
            self._advance_signal_timers()
            self._episode_arrived += traci.simulation.getArrivedNumber()
            self._episode_departed += traci.simulation.getDepartedNumber()
            if hasattr(traci.simulation, "getEmergencyStoppingVehiclesNumber"):
                self._episode_emergency_stops += (
                    traci.simulation.getEmergencyStoppingVehiclesNumber()
                )

        for ts_id in self.ts_ids:
            obs[ts_id] = self._get_observation(ts_id)

        return obs

    def _get_observation(self, ts_id: str) -> np.ndarray:
        """
        Get normalized observation for a traffic light agent.
        
        All features are scaled to [0, 1] or [-1, 1] for stable neural network training.
        This follows best practice from the proposal (Bab IV §4.2) and deep learning standards.

        Returns concatenated vector:
        [queue_norm, delta_queue_norm, density_norm, waiting_norm, 
         delta_density_norm, phase_onehot, phase_duration_norm]
        
        Feature ranges:
        - queue_norm:        [0, 1]
        - delta_queue_norm:  [-1, 1]
        - density_norm:      [0, 1]
        - waiting_norm:      [0, 1]
        - delta_density_norm: [-1, 1]
        - phase_onehot:      one-hot (binary)
        - phase_duration:    [0, 1]
        """
        lanes = self.controlled_lanes[ts_id]
        num_lanes = len(lanes)

        # === RAW FEATURES ===
        # Queue lengths per lane
        queue_raw = np.array(
            [traci.lane.getLastStepHaltingNumber(lane) for lane in lanes],
            dtype=np.float32,
        )

        # Density per lane (vehicles / lane_capacity)
        density_raw = np.array(
            [
                traci.lane.getLastStepVehicleNumber(lane)
                / max(traci.lane.getLength(lane) / 7.0, 1.0)  # ~7m per vehicle
                for lane in lanes
            ],
            dtype=np.float32,
        )

        # Waiting time per lane
        waiting_raw = np.array(
            [traci.lane.getWaitingTime(lane) for lane in lanes],
            dtype=np.float32,
        )

        # === NORMALIZED FEATURES (best practice: scale to [0, 1] or [-1, 1]) ===
        # Queue: normalize by max expected queue per lane
        queue_norm = queue_raw / self.max_queue_per_lane
        queue_norm = np.clip(queue_norm, 0.0, 1.0)

        # Density: already in [0, 1] from capacity calculation
        density_norm = np.clip(density_raw, 0.0, 1.0)

        # Waiting time: normalize by max expected waiting time
        waiting_norm = waiting_raw / self.max_waiting_time
        waiting_norm = np.clip(waiting_norm, 0.0, 1.0)

        # === TEMPORAL DELTAS ===
        delta_queue_raw = queue_raw - self._prev_queue[ts_id]
        delta_density_raw = density_raw - self._prev_density[ts_id]

        # Normalize deltas to [-1, 1] range
        delta_queue_norm = delta_queue_raw / self.max_delta_queue
        delta_queue_norm = np.clip(delta_queue_norm, -1.0, 1.0)

        delta_density_norm = delta_density_raw / self.max_delta_density
        delta_density_norm = np.clip(delta_density_norm, -1.0, 1.0)

        # Update previous values (store raw for next delta)
        self._prev_queue[ts_id] = queue_raw.copy()
        self._prev_density[ts_id] = density_raw.copy()

        # === PHASE FEATURES (already normalized) ===
        num_phases = self.num_phases[ts_id]
        phase_onehot = np.zeros(num_phases, dtype=np.float32)
        current_phase_idx = self._current_phase[ts_id]
        if current_phase_idx < num_phases:
            phase_onehot[current_phase_idx] = 1.0

        # Phase duration: normalize by max_green and clip to [0, 1]
        phase_duration_norm = self._phase_duration[ts_id] / self.max_green
        phase_duration_norm = float(np.clip(phase_duration_norm, 0.0, 1.0))

        # === CONCATENATE NORMALIZED OBSERVATION ===
        # [queue_norm, delta_queue_norm, density_norm, waiting_norm, 
        #  delta_density_norm, phase_onehot, phase_duration_norm]
        obs = np.concatenate(
            [
                queue_norm,           # [0, 1]
                delta_queue_norm,     # [-1, 1]
                density_norm,         # [0, 1]
                waiting_norm,         # [0, 1]
                delta_density_norm,    # [-1, 1]
                phase_onehot,         # one-hot
                [phase_duration_norm] # [0, 1]
            ]
        )

        return obs

    def _get_raw_observation(self, ts_id: str) -> Dict[str, np.ndarray]:
        """
        Get RAW (unnormalized) observation values for debugging/logging.
        
        Returns dict with named features instead of concatenated vector.
        """
        lanes = self.controlled_lanes[ts_id]
        
        queue = np.array(
            [traci.lane.getLastStepHaltingNumber(lane) for lane in lanes],
            dtype=np.float32,
        )
        density = np.array(
            [traci.lane.getLastStepVehicleNumber(lane) 
             / max(traci.lane.getLength(lane) / 7.0, 1.0) for lane in lanes],
            dtype=np.float32,
        )
        waiting = np.array(
            [traci.lane.getWaitingTime(lane) for lane in lanes],
            dtype=np.float32,
        )
        
        return {
            'queue': queue,
            'density': density,
            'waiting_time': waiting,
            'phase': self._current_phase[ts_id],
            'phase_duration': self._phase_duration[ts_id],
        }

    def get_obs_size(self, ts_id: str) -> int:
        """Get observation size for a specific traffic light."""
        num_lanes = len(self.controlled_lanes[ts_id])
        num_phases = self.num_phases[ts_id]
        # queue + delta_queue + density + waiting + delta_density + phase_onehot + duration
        return num_lanes * 5 + num_phases + 1

    def get_action_size(self, ts_id: str) -> int:
        """Get number of available actions (phases) for a traffic light."""
        return self.num_phases[ts_id]

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict]:
        """
        Execute actions and advance simulation.

        Args:
            actions: Dict mapping ts_id -> phase index

        Returns:
            observations, rewards, done, info
        """
        metrics_before = self._get_metrics()
        self._update_recovery_state(metrics_before)

        # Apply actions (with yellow transition / recovery override)
        if self._recovery_active:
            actions = self._build_recovery_actions()

        for ts_id, action in actions.items():
            self._apply_action(ts_id, action)

        # Simulate for delta_time steps
        for _ in range(self.delta_time):
            traci.simulationStep()
            self._step_count += 1
            self._advance_signal_timers()
            self._episode_arrived += traci.simulation.getArrivedNumber()
            self._episode_departed += traci.simulation.getDepartedNumber()
            if hasattr(traci.simulation, "getEmergencyStoppingVehiclesNumber"):
                self._episode_emergency_stops += (
                    traci.simulation.getEmergencyStoppingVehiclesNumber()
                )

        # Collect observations and rewards
        obs = {}
        rewards = {}
        for ts_id in self.ts_ids:
            obs[ts_id] = self._get_observation(ts_id)
            rewards[ts_id] = self._compute_reward(ts_id)

        done = self._step_count >= self.num_seconds

        info = {
            "step": self._step_count,
            "metrics": self._get_metrics(),
        }

        if done:
            traci.close()
            self._sumo_running = False

        return obs, rewards, done, info

    def _build_phase_lane_map(self, ts_id: str) -> List[List[str]]:
        """Precompute incoming lanes that receive green for each available phase."""
        controlled_links = traci.trafficlight.getControlledLinks(ts_id)
        phase_lanes: List[List[str]] = []
        for phase_state in self.phase_defs[ts_id]:
            lanes: Set[str] = set()
            for idx, link_group in enumerate(controlled_links):
                if idx >= len(phase_state):
                    break
                if phase_state[idx] not in ("G", "g"):
                    continue
                if not link_group:
                    continue
                first_link = link_group[0]
                if first_link and len(first_link) > 0:
                    lanes.add(first_link[0])
            phase_lanes.append(sorted(lanes))
        return phase_lanes

    def _select_phase_by_queue(self, ts_id: str, allow_current: bool) -> int:
        """Select phase serving the highest queued incoming lanes."""
        current = self._current_phase[ts_id]
        best_phase = current
        best_queue = -1.0
        for phase_idx, lanes in enumerate(self._phase_lanes[ts_id]):
            if not allow_current and phase_idx == current:
                continue
            queue_score = float(
                sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)
            )
            if queue_score > best_queue:
                best_queue = queue_score
                best_phase = phase_idx
        return best_phase

    def _build_recovery_actions(self) -> Dict[str, int]:
        """Use deterministic longest-queue phase selection during recovery mode."""
        actions: Dict[str, int] = {}
        for ts_id in self.ts_ids:
            actions[ts_id] = self._select_phase_by_queue(ts_id, allow_current=True)
        return actions

    def _update_recovery_state(self, metrics: Dict[str, float]):
        """Manage recovery-mode entry/exit using hysteresis counters."""
        if not self.recovery_enabled:
            self._recovery_active = False
            self._recovery_enter_counter = 0
            self._recovery_exit_counter = 0
            return

        enter_condition = (
            metrics["avg_delay"] > self.recovery_enter_delay
            or metrics["avg_queue"] > self.recovery_enter_queue
        )
        exit_condition = (
            metrics["avg_delay"] < self.recovery_exit_delay
            and metrics["avg_queue"] < self.recovery_exit_queue
        )

        if not self._recovery_active:
            self._recovery_enter_counter = (
                self._recovery_enter_counter + 1 if enter_condition else 0
            )
            if self._recovery_enter_counter >= self.recovery_hold_steps:
                self._recovery_active = True
                self._recovery_enter_counter = 0
                self._recovery_exit_counter = 0
            return

        self._recovery_exit_counter = (
            self._recovery_exit_counter + 1 if exit_condition else 0
        )
        if self._recovery_exit_counter >= self.recovery_hold_steps:
            self._recovery_active = False
            self._recovery_exit_counter = 0
            self._recovery_enter_counter = 0

    def _apply_action(self, ts_id: str, action: int):
        """Apply a phase action with yellow transition if needed."""
        current = self._current_phase[ts_id]

        # Ignore invalid actions to avoid entering yellow with no valid target phase.
        if not (0 <= action < len(self.phase_defs[ts_id])):
            return

        # Ignore new commands while a yellow transition is ongoing.
        if self._yellow_phase_active[ts_id]:
            return

        if self._phase_duration[ts_id] >= self.max_green and len(self.phase_defs[ts_id]) > 1:
            action = self._select_phase_by_queue(ts_id, allow_current=False)

        # Enforce minimum green time before allowing phase change.
        if self._phase_duration[ts_id] < self.min_green:
            return

        if action != current:
            # Set yellow phase (all 'y')
            current_state = traci.trafficlight.getRedYellowGreenState(ts_id)
            yellow_state = ""
            for char in current_state:
                if char in ("G", "g"):
                    yellow_state += "y"
                else:
                    yellow_state += char
            traci.trafficlight.setRedYellowGreenState(ts_id, yellow_state)

            # Schedule delayed switch to target green after yellow_time seconds.
            self._yellow_phase_active[ts_id] = True
            self._pending_phase[ts_id] = action
            self._yellow_remaining[ts_id] = self.yellow_time

    def _advance_signal_timers(self):
        """Progress yellow transitions and apply pending green phases when ready."""
        for ts_id in self.ts_ids:
            if not self._yellow_phase_active[ts_id]:
                self._phase_duration[ts_id] += 1
                continue

            self._yellow_remaining[ts_id] -= 1
            if self._yellow_remaining[ts_id] > 0:
                continue

            target_action = self._pending_phase[ts_id]
            if (
                target_action is not None
                and 0 <= target_action < len(self.phase_defs[ts_id])
            ):
                traci.trafficlight.setRedYellowGreenState(
                    ts_id, self.phase_defs[ts_id][target_action]
                )
                self._current_phase[ts_id] = target_action
                self._phase_duration[ts_id] = 0

            self._yellow_phase_active[ts_id] = False
            self._pending_phase[ts_id] = None
            self._yellow_remaining[ts_id] = 0

    def _compute_reward(self, ts_id: str) -> float:
        """
        Compute reward for an agent.
        r_i = -(sum(queue_l) + alpha * sum(waiting_l))
        """
        lanes = self.controlled_lanes[ts_id]
        total_queue = sum(
            traci.lane.getLastStepHaltingNumber(lane) for lane in lanes
        )
        total_waiting = sum(traci.lane.getWaitingTime(lane) for lane in lanes)

        reward = -(total_queue + self.reward_alpha * total_waiting)
        return reward

    def _get_metrics(self) -> Dict[str, float]:
        """Get global traffic metrics."""
        vehicles = traci.vehicle.getIDList()
        total_waiting = sum(traci.vehicle.getWaitingTime(v) for v in vehicles)
        avg_delay = total_waiting / max(len(vehicles), 1)

        # Queue across all controlled lanes
        all_lanes = []
        for lanes in self.controlled_lanes.values():
            all_lanes.extend(lanes)
        total_queue = sum(
            traci.lane.getLastStepHaltingNumber(lane) for lane in all_lanes
        )
        avg_queue = total_queue / max(len(all_lanes), 1)

        return {
            "avg_delay": avg_delay,
            "avg_queue": avg_queue,
            # Throughput is defined as cumulative completed/arrived vehicles per episode.
            "throughput": self._episode_arrived,
            "arrived_vehicles": self._episode_arrived,
            "departed_vehicles": self._episode_departed,
            "emergency_stops": self._episode_emergency_stops,
            "num_vehicles": len(vehicles),
            "recovery_active": float(self._recovery_active),
        }

    def close(self):
        """Close the SUMO simulation."""
        if self._sumo_running:
            traci.close()
            self._sumo_running = False

    @property
    def edge_index(self) -> np.ndarray:
        """Return edge index in COO format [2, num_edges] for PyG."""
        rows, cols = np.where(self.adjacency_matrix > 0)
        return np.stack([rows, cols], axis=0).astype(np.int64)
