import json
from typing import Any

import numpy as np
import yaml

from simulator.state import EchelonState, SimulatorState


class SupplyChainSimulator:
    """
    Four-echelon deterministic serial supply-chain simulator.
    Structure: Retailer -> Wholesaler -> Distributor -> Manufacturer.
    State: inventory, backlog, pipeline_inventory, demand_history, last_order.
    Demand: Generated at retailer via normal distribution, clipped at zero.
    Lead Times & Capacity: Handled explicitly per-step.
    Cost: holding, backlog, ordering calculated each step.
    Timestep sequence: Demand -> Fulfill & Order -> Ship (up to capacity) -> Pipeline advances -> Inventory updates -> Cost calculated.
    RNG Strategy: Single NumPy Generator strictly seeded.
    Determinism: Repeated runs with same seed are byte-identical.
    """

    def __init__(self, config_path: str, seed: int | None = None):
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.num_echelons = len(self.config["echelons"])
        if self.num_echelons != 4:
            raise ValueError("Expected exactly 4 modeled echelons")

        self.lead_times = self.config["lead_time"]
        if len(self.lead_times) != self.num_echelons:
            raise ValueError("Mismatch between echelons and lead_time length")
        for lt in self.lead_times:
            if lt < 0:
                raise ValueError("Lead times cannot be negative")

        self.capacity = self.config["capacity"]["units_per_step_per_echelon"]
        if self.capacity < 0:
            raise ValueError("Capacity cannot be negative")

        self.horizon = self.config["horizon"]
        if self.horizon <= 0:
            raise ValueError("Horizon must be positive")

        self.cost_holding = self.config["cost"]["holding"]
        self.cost_backlog = self.config["cost"]["backlog"]
        self.cost_ordering = self.config["cost"]["ordering"]
        if self.cost_holding < 0 or self.cost_backlog < 0 or self.cost_ordering < 0:
            raise ValueError("Cost coefficients cannot be negative")

        self.demand_mean = self.config["demand"]["mean"]
        self.demand_std = self.config["demand"]["std"]
        self.clip_demand = self.config["demand"]["clip_at_zero"]

        # Determine seed
        self._seed = seed if seed is not None else self.config.get("seed", None)
        if self._seed is None:
            self._seed = 42  # default fallback if not in config and not provided

        self.rng = np.random.default_rng(self._seed)
        self.current_step = 0
        self.state = None
        self.reset()

    def reset(self, seed: int | None = None) -> SimulatorState:
        if seed is not None:
            self._seed = seed
        self.rng = np.random.default_rng(self._seed)
        self.current_step = 0

        echelons = []
        for i in range(self.num_echelons):
            # pipeline length is exactly lead_time[i]
            pipeline = [0] * self.lead_times[i]
            echelons.append(
                EchelonState(
                    inventory=0,
                    backlog=0,
                    pipeline=pipeline,
                    pipeline_inventory=0,
                    demand_history=[],
                    last_order=0,
                )
            )

        self.state = SimulatorState(
            step=self.current_step,
            echelons=echelons,
            total_cost=0.0,
        )
        return self._get_state_copy()

    def _get_state_copy(self) -> SimulatorState:
        echelons_copy = [
            EchelonState(
                inventory=e.inventory,
                backlog=e.backlog,
                pipeline=e.pipeline.copy(),
                pipeline_inventory=e.pipeline_inventory,
                demand_history=e.demand_history.copy(),
                last_order=e.last_order,
            )
            for e in self.state.echelons
        ]
        return SimulatorState(
            step=self.state.step,
            echelons=echelons_copy,
            total_cost=self.state.total_cost,
        )

    def step(self, actions: list[int]) -> tuple[SimulatorState, dict[str, Any], bool]:
        """
        Transition semantics within a step:
        1. demand (retailer generates external demand, others receive order from downstream as demand)
        2. retailer demand fulfillment/backlog + upstream orders processing
        3. upstream orders (echelons place orders action[i] to i+1)
        4. order processing (echelon i+1 processes the order action[i] + its own backlog, and ships to i)
        5. shipment pipeline (shipped amounts enter pipeline)
        6. delayed arrivals (pipeline items that reach 0 delay arrive)
        7. inventory updates (inventory += arrivals)
        8. cost (computed on ending inventory and backlog)
        """
        if len(actions) != self.num_echelons:
            raise ValueError(
                f"Expected {self.num_echelons} actions, got {len(actions)}"
            )

        for a in actions:
            if not isinstance(a, int) or a < 0:
                raise ValueError("Actions must be non-negative integers")

        # 1. Demand generation
        base_demand = self.rng.normal(self.demand_mean, self.demand_std)
        
        # Add trend
        if "trend" in self.config.get("demand", {}):
            base_demand += self.config["demand"]["trend"] * self.current_step
            
        # Add seasonality
        if "seasonality" in self.config.get("demand", {}):
            period = self.config["demand"]["seasonality"].get("period", 12)
            amplitude = self.config["demand"]["seasonality"].get("amplitude", 10)
            base_demand += amplitude * np.sin(2 * np.pi * self.current_step / period)
            
        # Add spike/shock
        if "spike" in self.config:
            if isinstance(self.config["spike"], dict):
                if self.current_step == self.config["spike"].get("step", -1):
                    base_demand += self.config["spike"].get("magnitude", 0)
            elif isinstance(self.config["spike"], list):
                for spike in self.config["spike"]:
                    if self.current_step == spike.get("step", -1):
                        base_demand += spike.get("magnitude", 0)

        customer_demand = int(np.round(base_demand))
        
        if self.clip_demand and customer_demand < 0:
            customer_demand = 0

        demands = [0] * self.num_echelons
        demands[0] = customer_demand
        for i in range(1, self.num_echelons):
            demands[i] = actions[i - 1]

        # 2 & 3 & 4. Demand fulfillment & Order processing
        shipped = [0] * (
            self.num_echelons + 1
        )  # indices 0..3 for modeled, 4 for supplier
        for i in range(self.num_echelons):
            e = self.state.echelons[i]
            e.demand_history.append(demands[i])
            e.last_order = actions[i]

            total_requested = demands[i] + e.backlog
            can_ship = min(total_requested, e.inventory, self.capacity)
            shipped[i] = can_ship

            e.inventory -= can_ship
            e.backlog = total_requested - can_ship

        # Supplier processes order from manufacturer (infinite capacity and inventory)
        shipped[self.num_echelons] = actions[-1]

        # 5. Shipment pipeline & 6. Delayed arrivals & 7. Inventory updates
        for i in range(self.num_echelons):
            e = self.state.echelons[i]
            incoming_shipment = shipped[i + 1]

            e.pipeline.append(incoming_shipment)
            arriving = e.pipeline.pop(0)

            e.inventory += arriving
            e.pipeline_inventory = sum(e.pipeline)

        # 8. Cost calculation
        step_cost = 0.0
        for i in range(self.num_echelons):
            e = self.state.echelons[i]
            c_h = e.inventory * self.cost_holding
            c_b = e.backlog * self.cost_backlog
            c_o = actions[i] * self.cost_ordering
            step_cost += c_h + c_b + c_o

        self.current_step += 1
        self.state.step = self.current_step
        self.state.total_cost += step_cost

        done = self.current_step >= self.horizon

        info = {
            "demand": demands,
            "shipped": shipped[:-1],
            "step_cost": step_cost,
        }

        return self._get_state_copy(), info, done

    def serialize_trajectory(self, actions_sequence: list[list[int]]) -> bytes:
        """
        Produce a deterministic byte representation of the state trajectory
        for the given action sequence. Used for the determinism test.
        """
        trajectory = []
        trajectory.append(self.state.get_dict())
        for actions in actions_sequence:
            state, _, _ = self.step(actions)
            trajectory.append(state.get_dict())

        return json.dumps(trajectory, sort_keys=True).encode("utf-8")
