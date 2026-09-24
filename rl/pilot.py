import json
import os
import time

import numpy as np
import yaml

from baselines.out import OUTPolicy
from rl.pilot_env import PilotOneEchelonEnv
from rl.trainer import IPPOTrainer


class SyncPilotVectorEnv:
    def __init__(self, num_envs: int, config_path: str | None = None):
        self.num_envs = num_envs
        
        observation_mode = "aggregate"
        if config_path:
            with open(config_path, "r") as f:
                import yaml
                cfg = yaml.safe_load(f)
                observation_mode = cfg.get("observation_mode", "aggregate")
                
        self.envs = [PilotOneEchelonEnv(observation_mode=observation_mode) for _ in range(num_envs)]
        self.agents = self.envs[0].possible_agents

    def reset(self, seed: int | None = None):
        obs_list = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=seed + i if seed is not None else None)
            obs_list.append(obs)
        return obs_list

    def step(self, actions_list):
        obs_list, rews_list, terms_list, truncs_list, infos_list = [], [], [], [], []
        for i, env in enumerate(self.envs):
            obs, rews, terms, truncs, infos = env.step(actions_list[i])
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


class PilotTrainer(IPPOTrainer):
    def __init__(self, config_path: str):
        import rl.trainer

        original = rl.trainer.SyncVectorEnv
        rl.trainer.SyncVectorEnv = lambda num_envs, c_path: SyncPilotVectorEnv(
            num_envs, config_path
        )
        try:
            super().__init__(config_path)
        finally:
            rl.trainer.SyncVectorEnv = original


def run_pilot(updates=50, seeds=None, output_file="pilot_convergence_500.json"):
    if seeds is None:
        seeds = [0, 1, 2]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(repo_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    pilot_config_path = os.path.join(
        repo_root, "configs", "phase4_1_echelon_pilot.yaml"
    )

    with open(os.path.join(repo_root, "configs", "phase4_ippo.yaml"), "r") as f:
        config = yaml.safe_load(f)
    config["simulator_config"] = "configs/phase1_simulator.yaml"  # Dummy
    with open(pilot_config_path, "w") as f:
        yaml.dump(config, f)

    # 1. OUT Reference
    env = PilotOneEchelonEnv()
    out_costs_per_seed = {}

    z_scores = [0.5]
    policy = OUTPolicy(z_scores=z_scores, lead_times=[2], capacity=100)

    for seed in seeds:
        env.reset(seed=seed)
        done = False
        total_cost = 0.0
        while not done:
            fake_state = env.get_fake_simulator_state()
            actions_list = policy.get_actions(fake_state)
            action = actions_list[0]

            _, rews, terms, truncs, _ = env.step({"retailer": action})
            total_cost += -rews["retailer"]
            done = terms["retailer"] or truncs["retailer"]
        out_costs_per_seed[seed] = total_cost

    mean_out_cost = np.mean(list(out_costs_per_seed.values()))

    # 2. Train PPO
    results = {"seeds": {}, "out_reference_aggregate": mean_out_cost, "config": config}

    for seed in seeds:
        trainer = PilotTrainer(pilot_config_path)
        trainer.seed = seed

        best_cost = float("inf")
        final_cost = float("inf")
        best_update = 0

        trajectory = []

        start_time = time.time()

        for update in range(1, updates + 1):
            metrics = trainer.train(1, use_wandb=False)
            cost = metrics["total_cost"]

            if cost < best_cost:
                best_cost = cost
                best_update = update

            final_cost = cost
            trajectory.append(
                {
                    "update": update,
                    "cost": cost,
                    "policy_loss": metrics.get("retailer_policy_loss", 0),
                    "value_loss": metrics.get("retailer_value_loss", 0),
                    "entropy": metrics.get("retailer_entropy", 0),
                    "elapsed_time": time.time() - start_time,
                }
            )

            # Save intermediate results in case of crash
            results["seeds"][seed] = {
                "out_cost": out_costs_per_seed[seed],
                "initial_cost": trajectory[0]["cost"],
                "best_cost": best_cost,
                "final_cost": final_cost,
                "best_update": best_update,
                "total_updates": updates,
                "wall_clock_time": time.time() - start_time,
                "trajectory": trajectory,
            }
            with open(os.path.join(results_dir, output_file), "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_pilot()
