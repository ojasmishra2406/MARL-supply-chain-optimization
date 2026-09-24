from typing import Any

import numpy as np

from envs.supply_chain_env import SupplyChainParallelEnv


class SyncVectorEnv:
    """Minimal synchronous vector environment for PettingZoo ParallelEnv."""

    def __init__(self, num_envs: int, config_path: str, comm_enabled: bool = False):
        self.num_envs = num_envs
        self.envs = [SupplyChainParallelEnv(config_path, comm_enabled=comm_enabled) for _ in range(num_envs)]
        self.agents = self.envs[0].possible_agents

    def reset(self, seed: int | None = None) -> list[dict[str, np.ndarray]]:
        # Seed the first environment, others increment or use rng
        # But to be deterministic, we seed them with seed + i
        obs_list = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=seed + i if seed is not None else None)
            obs_list.append(obs)
        return obs_list

    def step(self, actions_list: list[dict[str, Any]]) -> tuple[
        list[dict[str, np.ndarray]],
        list[dict[str, float]],
        list[dict[str, bool]],
        list[dict[str, bool]],
        list[dict[str, Any]],
    ]:
        obs_list = []
        rews_list = []
        terms_list = []
        truncs_list = []
        infos_list = []

        for i, env in enumerate(self.envs):
            obs, rews, terms, truncs, infos = env.step(actions_list[i])

            # If the environment is done for all agents (which happens at horizon)
            env_done = (
                all(terms.values()) or all(truncs.values()) or len(env.agents) == 0
            )

            if env_done:
                infos["final_observation"] = obs
                obs, _ = env.reset()

            obs_list.append(obs)
            rews_list.append(rews)
            terms_list.append(terms)
            truncs_list.append(truncs)
            infos_list.append(infos)

        return obs_list, rews_list, terms_list, truncs_list, infos_list
