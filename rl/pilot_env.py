from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from simulator.state import EchelonState, SimulatorState


class PilotOneEchelonEnv(ParallelEnv):
    metadata = {"render_modes": [], "name": "PilotOneEchelon-v0"}  # noqa: RUF012

    def __init__(self, config_path: str | None = None, observation_mode: str = "aggregate"):
        super().__init__()
        self.possible_agents = ["retailer"]
        self.agents = ["retailer"]

        self.capacity = 100
        self.lead_time = 2
        self.horizon = 52

        self.cost_holding = 1.0
        self.cost_backlog = 2.0
        self.cost_ordering = 0.5

        self.mu = 20.0
        self.sigma = 5.0
        self.observation_mode = observation_mode

        obs_dim = 5 if observation_mode == "aggregate" else 4 + self.lead_time
        self.observation_spaces = {
            "retailer": spaces.Box(low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        }
        self.action_spaces = {"retailer": spaces.Discrete(self.capacity + 1)}

        self.rng = np.random.default_rng()
        self.step_count = 0

        # State tracking
        self.inventory = 0.0
        self.backlog = 0.0
        self.pipeline = []  # length 2
        self.last_demand = 0.0
        self.last_order = 0.0

        self.total_cost = 0.0

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agents = self.possible_agents.copy()
        self.step_count = 0

        self.inventory = 0.0
        self.backlog = 0.0
        self.pipeline = [0.0] * self.lead_time
        self.last_demand = 0.0
        self.last_order = 0.0
        self.total_cost = 0.0

        obs = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        return obs, infos

    def _get_observations(self) -> dict[str, Any]:
        if self.observation_mode == "aggregate":
            pipeline_inv = float(sum(self.pipeline))
            base_obs = np.array(
                [
                    float(self.inventory),
                    float(self.backlog),
                    pipeline_inv,
                    float(self.last_demand),
                    float(self.last_order),
                ],
                dtype=np.float32,
            )
        else:
            base_obs = np.array(
                [
                    float(self.inventory),
                    float(self.backlog),
                ] + [float(p) for p in self.pipeline] + [
                    float(self.last_demand),
                    float(self.last_order),
                ],
                dtype=np.float32,
            )
        return {"retailer": base_obs}

    def get_fake_simulator_state(self) -> SimulatorState:
        # Used for evaluating OUT baseline
        echelon_state = EchelonState(
            inventory=int(self.inventory),
            backlog=int(self.backlog),
            pipeline_inventory=int(sum(self.pipeline)),
            demand_history=[int(self.last_demand)] if self.last_demand > 0 else [],
            last_order=int(self.last_order),
        )
        # Mock step since it's required for SimulatorState instantiation
        return SimulatorState(
            step=self.step_count, total_cost=self.total_cost, echelons=[echelon_state]
        )

    def step(self, actions: dict[str, Any]) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        if not self.agents:
            return {}, {}, {}, {}, {}

        # 1. Demand realization
        demand = self.rng.normal(self.mu, self.sigma)
        demand = max(0.0, float(round(demand)))
        self.last_demand = demand

        # 2. Receive shipments
        arriving = self.pipeline.pop(0) if self.pipeline else 0.0
        self.inventory += arriving

        # 3. Satisfy demand
        total_demand = demand + self.backlog
        fulfilled = min(self.inventory, total_demand)
        self.inventory -= fulfilled
        self.backlog = total_demand - fulfilled

        # 4. Process order
        action = float(actions.get("retailer", 0.0))
        # No upstream limit (infinite supplier), so order is always fully accepted
        self.pipeline.append(action)
        self.last_order = action

        # 5. Cost calculation
        cost_h = self.inventory * self.cost_holding
        cost_b = self.backlog * self.cost_backlog
        cost_o = action * self.cost_ordering
        step_cost = cost_h + cost_b + cost_o
        self.total_cost += step_cost

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.horizon

        obs = self._get_observations()
        rewards = {"retailer": -float(step_cost)}
        terminations = {"retailer": terminated}
        truncations = {"retailer": truncated}
        infos = {"retailer": {
            "demand": demand,
            "fulfilled": fulfilled,
            "order": action,
            "cost": step_cost
        }}

        if terminated or truncated:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]
