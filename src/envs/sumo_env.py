"""
SUMO Multi-Agent Traffic Signal Control Environment.
Wraps SUMO simulator via TraCI for GNN-MARL training.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional

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
        route_file: str,
        num_seconds: int = 3600,
        delta_time: int = 5,
        yellow_time: int = 3,
        min_green: int = 10,
        max_green: int = 60,
        reward_alpha: float = 0.5,
        use_gui: bool = False,
        seed: int = 42,
        # Normalization parameters (best practice from GAP_AUDIT)
        max_queue_per_lane: int = 30,      # Max vehicles that can queue per lane
        max_waiting_time: float = 300.0,   # Max waiting time in seconds (5 min)
        max_delta_queue: float = 10.0,     # Max expected queue change per step
        max_delta_density: float = 0.5,    # Max expected density change per step
    ):
        self.net_file = net_file
        self.route_file = route_file
        self.num_seconds = num_seconds
        self.delta_time = delta_time  # seconds between agent decisions
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.reward_alpha = reward_alpha
        self.use_gui = use_gui
        self.seed = seed

        # Normalization constants
        self.max_queue_per_lane = max_queue_per_lane
        self.max_waiting_time = max_waiting_time
        self.max_delta_queue = max_delta_queue
        self.max_delta_density = max_delta_density

        # Load network to get topology info
        self.net = sumolib.net.readNet(self.net_file)

        # Get traffic light IDs
        self.ts_ids: List[str] = [tl.getID() for tl in self.net.getTrafficLights()]
        self.num_agents = len(self.ts_ids)

        # Build adjacency info from network topology
        self.adjacency_matrix = self._build_adjacency()

        # Per-agent info
        self.controlled_lanes: Dict[str, List[str]] = {}
        self.num_phases: Dict[str, int] = {}
        self.phase_defs: Dict[str, List[str]] = {}

        # State tracking for temporal features
        self._prev_queue: Dict[str, np.ndarray] = {}
        self._prev_density: Dict[str, np.ndarray] = {}

        # Timing
        self._step_count = 0
        self._yellow_phase_active: Dict[str, bool] = {}
        self._current_phase: Dict[str, int] = {}
        self._phase_duration: Dict[str, int] = {}

        self._sumo_running = False

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
        return [
            sumo_path,
            "-n", self.net_file,
            "-r", self.route_file,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1",
            "--seed", str(self.seed),
        ]

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment and return initial observations."""
        if self._sumo_running:
            traci.close()

        traci.start(self._get_sumo_cmd())
        self._sumo_running = True
        self._step_count = 0

        # Initialize per-TL info from running simulation
        for ts_id in self.ts_ids:
            self.controlled_lanes[ts_id] = list(
                set(traci.trafficlight.getControlledLanes(ts_id))
            )
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

            self._current_phase[ts_id] = 0
            self._phase_duration[ts_id] = 0
            self._yellow_phase_active[ts_id] = False

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

        # Phase duration: normalize by max_green
        phase_duration_norm = self._phase_duration[ts_id] / self.max_green

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
        # Apply actions (with yellow transition)
        for ts_id, action in actions.items():
            self._apply_action(ts_id, action)

        # Simulate for delta_time steps
        for _ in range(self.delta_time):
            traci.simulationStep()
            self._step_count += 1

        # Update phase durations
        for ts_id in self.ts_ids:
            self._phase_duration[ts_id] += self.delta_time

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

    def _apply_action(self, ts_id: str, action: int):
        """Apply a phase action with yellow transition if needed."""
        current = self._current_phase[ts_id]

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

            # After yellow_time, set new green phase
            # (simplified: we set green immediately after yellow in next decision step)
            self._current_phase[ts_id] = action
            self._phase_duration[ts_id] = 0

            # Set the target green phase
            if action < len(self.phase_defs[ts_id]):
                traci.trafficlight.setRedYellowGreenState(
                    ts_id, self.phase_defs[ts_id][action]
                )

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
        if len(vehicles) == 0:
            return {"avg_delay": 0.0, "avg_queue": 0.0, "throughput": 0}

        total_waiting = sum(traci.vehicle.getWaitingTime(v) for v in vehicles)
        avg_delay = total_waiting / len(vehicles)

        # Queue across all controlled lanes
        all_lanes = []
        for lanes in self.controlled_lanes.values():
            all_lanes.extend(lanes)
        total_queue = sum(
            traci.lane.getLastStepHaltingNumber(lane) for lane in all_lanes
        )
        avg_queue = total_queue / max(len(all_lanes), 1)

        departed = traci.simulation.getDepartedNumber()

        return {
            "avg_delay": avg_delay,
            "avg_queue": avg_queue,
            "throughput": departed,
            "num_vehicles": len(vehicles),
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
