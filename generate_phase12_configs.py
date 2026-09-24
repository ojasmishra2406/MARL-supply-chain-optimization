import os
import yaml

configs = {
    "primary_baselines.yaml": {
        "ablation_family": "Baseline",
        "algorithm": ["ippo", "mappo"],
        "architecture": "mlp",
        "forecasting": {"enabled": False},
        "gnn": {"enabled": False},
        "communication": {"enabled": False},
        "centralized_critic": [False, True],
        "regime": ["baseline", "high_variance"],
        "seed": [0, 1, 2, 3, 4],
        "environment": "configs/phase1_simulator.yaml",
        "evaluation_settings": {
            "scenarios": ["in_distribution", "demand_shift", "demand_spike", "lead_time_shift", "capacity_disruption", "combined_shift"],
            "eval_seeds": [0, 1, 2, 3, 4]
        }
    },
    "communication_ablation.yaml": {
        "ablation_family": "C_Communication",
        "algorithm": ["ippo_comm", "mappo_comm"],
        "architecture": "mlp",
        "forecasting": {"enabled": False},
        "gnn": {"enabled": False},
        "communication": {"enabled": True},
        "centralized_critic": [False, True],
        "regime": ["baseline", "high_variance"],
        "seed": [0, 1, 2, 3, 4],
        "environment": "configs/phase1_simulator.yaml",
        "evaluation_settings": {
            "scenarios": ["in_distribution", "demand_shift", "demand_spike", "lead_time_shift", "capacity_disruption", "combined_shift"],
            "eval_seeds": [0, 1, 2, 3, 4]
        }
    },
    "critic_ablation.yaml": {
        "ablation_family": "D_Critic",
        "algorithm": ["ippo", "mappo"],
        "architecture": "mlp",
        "forecasting": {"enabled": False},
        "gnn": {"enabled": False},
        "communication": {"enabled": False},
        "centralized_critic": [False, True],
        "regime": ["baseline", "high_variance"],
        "seed": [0, 1, 2, 3, 4],
        "environment": "configs/phase1_simulator.yaml",
        "evaluation_settings": {
            "scenarios": ["in_distribution", "demand_shift", "demand_spike", "lead_time_shift", "capacity_disruption", "combined_shift"],
            "eval_seeds": [0, 1, 2, 3, 4]
        }
    },
    "forecast_model_ablation.yaml": {
        "ablation_family": "E_Forecast_Model",
        "algorithm": "mappo",
        "architecture": "mlp",
        "forecasting": {"enabled": True, "model": ["naive", "xgboost"], "horizon": 2},
        "gnn": {"enabled": False},
        "communication": {"enabled": False},
        "centralized_critic": True,
        "regime": ["baseline", "high_variance"],
        "seed": [0, 1, 2, 3, 4],
        "environment": "configs/phase1_simulator.yaml",
        "evaluation_settings": {
            "scenarios": ["in_distribution", "demand_shift", "demand_spike", "lead_time_shift", "capacity_disruption", "combined_shift"],
            "eval_seeds": [0, 1, 2, 3, 4]
        }
    },
    "combined_ablation.yaml": {
        "ablation_family": "B_Forecasting_Combined",
        "algorithm": "mappo",
        "architecture": "gnn",
        "forecasting": {"enabled": True, "model": "ma", "horizon": 2},
        "gnn": {"enabled": True},
        "communication": {"enabled": False},
        "centralized_critic": True,
        "regime": ["baseline", "high_variance"],
        "seed": [0, 1, 2, 3, 4],
        "environment": "configs/phase1_simulator.yaml",
        "evaluation_settings": {
            "scenarios": ["in_distribution", "demand_shift", "demand_spike", "lead_time_shift", "capacity_disruption", "combined_shift"],
            "eval_seeds": [0, 1, 2, 3, 4]
        }
    }
}

os.makedirs("configs/phase12", exist_ok=True)
for filename, config in configs.items():
    with open(f"configs/phase12/{filename}", "w") as f:
        yaml.dump(config, f, sort_keys=False)
print("Wrote all configs.")
