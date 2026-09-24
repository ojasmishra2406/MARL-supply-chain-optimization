import copy
from typing import Any, Dict, List, Optional, Tuple

import gymnasium.spaces as spaces
import numpy as np
from pettingzoo import ParallelEnv

from simulator.core import SupplyChainSimulator


class SupplyChainParallelEnv(ParallelEnv):
    metadata = {"render_modes": [], "name": "SupplyChain-v0"}

    def __init__(self, config_path: str, comm_enabled: bool = False, comm_dim: int = 4):
        super().__init__()
        self.config_path = config_path
        self.comm_enabled = comm_enabled
        self.comm_dim = comm_dim
        self.render_mode = None

        self.simulator = SupplyChainSimulator(config_path)

        self.possible_agents = ["retailer", "wholesaler", "distributor", "manufacturer"]
        self.agents = self.possible_agents.copy()

        self._agent_indices = {agent: i for i, agent in enumerate(self.possible_agents)}

        # Capacity from simulator for action bound
        self.capacity = self.simulator.capacity
        self.horizon = self.simulator.horizon

        self.observation_spaces = dict()
        self.action_spaces = dict()

        for agent in self.possible_agents:
            # Action space
            base_action_space = spaces.Discrete(self.capacity + 1)
            if self.comm_enabled:
                self.action_spaces[agent] = spaces.Dict(
                    {
                        "action": base_action_space,
                        "message": spaces.Box(
                            low=-np.inf, high=np.inf, shape=(self.comm_dim,), dtype=np.float32
                        ),
                    }
                )
            else:
                self.action_spaces[agent] = base_action_space

            # Observation space
            base_obs_space = spaces.Box(
                low=0.0, high=np.inf, shape=(5,), dtype=np.float32
            )  # [inventory, backlog, pipeline_inventory, last_demand, last_order]

            if self.comm_enabled:
                # Upstream and downstream messages
                self.observation_spaces[agent] = spaces.Dict(
                    {
                        "state": base_obs_space,
                        "message_upstream": spaces.Box(
                            low=-np.inf, high=np.inf, shape=(self.comm_dim,), dtype=np.float32
                        ),
                        "message_downstream": spaces.Box(
                            low=-np.inf, high=np.inf, shape=(self.comm_dim,), dtype=np.float32
                        ),
                    }
                )
            else:
                self.observation_spaces[agent] = base_obs_space

        self.messages_in_transit: Dict[str, Dict[str, np.ndarray]] = {
            agent: {"up": np.zeros(self.comm_dim, dtype=np.float32), "down": np.zeros(self.comm_dim, dtype=np.float32)}
            for agent in self.possible_agents
        }
        self.step_count = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.agents = self.possible_agents.copy()
        self.simulator.reset(seed=seed)
        self.step_count = 0

        self.messages_in_transit = {
            agent: {"up": np.zeros(self.comm_dim, dtype=np.float32), "down": np.zeros(self.comm_dim, dtype=np.float32)}
            for agent in self.possible_agents
        }

        obs = self._get_observations()
        infos = {agent: {} for agent in self.agents}

        return obs, infos

    def _get_observations(self) -> Dict[str, Any]:
        obs = {}
        state = self.simulator.state
        for i, agent in enumerate(self.agents):
            e = state.echelons[i]
            last_demand = float(e.demand_history[-1]) if e.demand_history else 0.0
            base_obs = np.array(
                [
                    float(e.inventory),
                    float(e.backlog),
                    float(e.pipeline_inventory),
                    last_demand,
                    float(e.last_order),
                ],
                dtype=np.float32,
            )

            if self.comm_enabled:
                obs[agent] = {
                    "state": base_obs,
                    "message_upstream": self.messages_in_transit[agent]["up"].copy(),
                    "message_downstream": self.messages_in_transit[agent]["down"].copy(),
                }
            else:
                obs[agent] = base_obs
        return obs

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        if not self.agents:
            return {}, {}, {}, {}, {}

        # 1. Parse actions
        sim_actions = [0] * len(self.possible_agents)
        new_messages: Dict[str, Dict[str, np.ndarray]] = {
            agent: {"up": np.zeros(self.comm_dim, dtype=np.float32), "down": np.zeros(self.comm_dim, dtype=np.float32)}
            for agent in self.possible_agents
        }

        for i, agent in enumerate(self.possible_agents):
            agent_act = actions.get(agent)
            if agent_act is None:
                # If agent not in actions (e.g. already done), default to 0
                sim_actions[i] = 0
                continue

            if self.comm_enabled:
                sim_actions[i] = int(agent_act["action"])
                msg = np.array(agent_act["message"], dtype=np.float32)

                # Send message upstream and downstream based on topology
                # Retailer (0) sends to Wholesaler (1)
                # Wholesaler (1) sends to Retailer (0) and Distributor (2)
                # Distributor (2) sends to Wholesaler (1) and Manufacturer (3)
                # Manufacturer (3) sends to Distributor (2)

                if i < len(self.possible_agents) - 1:
                    # send to downstream (which is upstream in supply chain... wait. Retailer is downstream of Wholesaler.
                    # Retailer sends upstream TO wholesaler.
                    # So Retailer (i) sends UPSTREAM to i+1.
                    new_messages[self.possible_agents[i + 1]]["down"] = msg.copy()

                if i > 0:
                    # send to downstream (i-1)
                    new_messages[self.possible_agents[i - 1]]["up"] = msg.copy()

            else:
                sim_actions[i] = int(agent_act)

        # 2. Simulator step
        prev_cost = self.simulator.state.total_cost
        _, info, _ = self.simulator.step(sim_actions)
        new_cost = self.simulator.state.total_cost

        # 3. Update messages
        if self.comm_enabled:
            self.messages_in_transit = new_messages

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.horizon

        # 4. Generate returns
        obs = self._get_observations()
        rewards = {}
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        for i, agent in enumerate(self.agents):
            # Reward is negative cost incurred by the agent this step.
            e = self.simulator.state.echelons[i]
            # The cost is holding + backlog + ordering
            # Reconstruct agent cost:
            c_h = e.inventory * self.simulator.cost_holding
            c_b = e.backlog * self.simulator.cost_backlog
            c_o = sim_actions[i] * self.simulator.cost_ordering
            agent_cost = c_h + c_b + c_o
            rewards[agent] = -float(agent_cost)

        if terminated or truncated:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]
