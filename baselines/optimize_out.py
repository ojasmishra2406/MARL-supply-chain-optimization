import json
import os
from typing import Any

import yaml

from baselines.metrics import calculate_bullwhip, calculate_fill_rate
from baselines.out import OUTPolicy
from simulator.core import SupplyChainSimulator


def run_episode(
    sim: SupplyChainSimulator, policy: OUTPolicy, seed: int
) -> dict[str, Any]:
    state = sim.reset(seed=seed)
    backlogs_retailer = []
    orders_hist = [[] for _ in range(sim.num_echelons)]
    demands_hist = [[] for _ in range(sim.num_echelons)]

    total_cost = 0.0
    for _ in range(sim.horizon):
        actions = policy.get_actions(state)
        state, info, done = sim.step(actions)
        total_cost = state.total_cost
        backlogs_retailer.append(state.echelons[0].backlog)

        for i in range(sim.num_echelons):
            orders_hist[i].append(actions[i])
            demands_hist[i].append(info["demand"][i])

        if done:
            break

    fill_rate = calculate_fill_rate(backlogs_retailer)
    bullwhips = []
    for i in range(sim.num_echelons):
        bw = calculate_bullwhip(orders_hist[i], demands_hist[i])
        bullwhips.append(bw)

    return {"cost": total_cost, "fill_rate": fill_rate, "bullwhips": bullwhips}


def evaluate_z_scores(
    sim: SupplyChainSimulator, z_scores: list[float], seeds: list[int]
) -> dict[str, Any]:
    policy = OUTPolicy(
        z_scores=z_scores,
        lead_times=sim.config["lead_time"],
        capacity=sim.config["capacity"]["units_per_step_per_echelon"],
    )
    costs, fill_rates, bullwhips_list = [], [], []
    for seed in seeds:
        res = run_episode(sim, policy, seed)
        costs.append(res["cost"])
        fill_rates.append(res["fill_rate"])
        bullwhips_list.append(res["bullwhips"])

    mean_cost = sum(costs) / len(costs)
    mean_fill_rate = sum(fill_rates) / len(fill_rates)

    mean_bullwhips = []
    for i in range(sim.num_echelons):
        mean_bw = sum(b[i] for b in bullwhips_list) / len(bullwhips_list)
        mean_bullwhips.append(mean_bw)

    return {
        "mean_cost": mean_cost,
        "costs": costs,
        "mean_fill_rate": mean_fill_rate,
        "mean_bullwhips": mean_bullwhips,
    }


def optimize_out_coordinate_descent(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sim_path = os.path.join(repo_root, config["simulator_config"])
    sim = SupplyChainSimulator(sim_path)

    seeds = config["seeds"]
    grid = config["z_score_grid"]
    rounds = config["rounds"]

    z_vector = [grid[len(grid) // 2]] * sim.num_echelons
    trace = []

    for r in range(rounds):
        round_trace = {}
        for echelon_idx in range(sim.num_echelons):
            best_cost = float("inf")
            best_z = grid[0]
            coord_trace = []
            for z in grid:
                cand = list(z_vector)
                cand[echelon_idx] = z
                res = evaluate_z_scores(sim, cand, seeds)
                cost = res["mean_cost"]
                coord_trace.append({"z": z, "cost": cost})

                if cost < best_cost:
                    best_cost = cost
                    best_z = z
            z_vector[echelon_idx] = best_z
            round_trace[f"echelon_{echelon_idx}"] = {
                "selected_z": best_z,
                "candidates": coord_trace,
            }
        trace.append({f"round_{r+1}": round_trace})

    final_res = evaluate_z_scores(sim, z_vector, seeds)
    return {
        "phase": 3,
        "algorithm": "Coordinate Descent OUT Baseline",
        "configuration_reference": config_path,
        "seed_list": seeds,
        "z_score_grid": grid,
        "rounds": rounds,
        "optimized_z_score_vector": z_vector,
        "per_seed_costs": final_res["costs"],
        "mean_total_cost": final_res["mean_cost"],
        "service_level": final_res["mean_fill_rate"],
        "bullwhips": final_res["mean_bullwhips"],
        "trace": trace,
    }


if __name__ == "__main__":
    import time

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(repo_root, "configs", "phase3_out.yaml")
    start = time.time()
    res = optimize_out_coordinate_descent(config_path)
    res["runtime_seconds"] = time.time() - start
    print(json.dumps(res, indent=2))
    with open(os.path.join(repo_root, "configs", "phase3_result.json"), "w") as f:
        json.dump(res, f, indent=2)
