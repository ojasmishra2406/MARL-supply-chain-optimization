# Configuration for Phase 8 Analysis

# Predefined Research Question Families
# Each family will have FDR correction applied independently.
RESEARCH_FAMILIES = {
    "algorithm_comparison": [
        ("ippo", "mappo"),
        ("mappo", "out"),
    ],
    "scenario_robustness": [
        # Evaluating MAPPO across scenarios compared to in_distribution
        ("in_distribution", "demand_shift"),
        ("in_distribution", "lead_time_shift"),
        ("in_distribution", "capacity_disruption"),
        ("in_distribution", "demand_spike"),
        ("in_distribution", "combined_shift"),
    ],
    "ablations": [
        ("mappo", "ablation3_communication"), # Communication
        ("mappo", "ablation2_parameter_sharing"), # Parameter sharing
        ("mappo", "ablation1_centralized"), # Critic ablation
        # Note: reward-weighting ablation (ablation5_high_backlog_cost) is also a comparison
        ("mappo", "ablation5_high_backlog_cost"),
    ]
}

# Thresholds
EFFECT_SIZE_THRESHOLD = 0.5  # |d| >= 0.5 is practically significant
BOOTSTRAP_RESAMPLES = 1000
ALPHA = 0.05
