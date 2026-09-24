import json
from .config import RESEARCH_FAMILIES
from .statistics import compare_groups
from .effect_sizes import cohens_d, is_practically_significant
from .corrections import benjamini_hochberg_correction
from .bootstrap import bootstrap_ci

def extract_seed_level_data(valid_results, target_algo, target_scenario, metric):
    """
    Extracts the seed-level aggregate of a metric.
    This enforces the distinction between episode-level uncertainty and training-seed-level robustness.
    For Phase 7, each result dictionary typically represents one seed's evaluation.
    If it represents multiple episodes per seed, we must average them to get 1 value per seed.
    """
    seed_values = {}
    for r in valid_results:
        if r["algorithm"] == target_algo and r["scenario"] == target_scenario:
            seed = r["seed"]
            if seed not in seed_values:
                seed_values[seed] = []
            seed_values[seed].append(r[metric])
            
    # Average across episodes to get seed-level data
    seed_aggregates = []
    for seed, values in sorted(seed_values.items()):
        seed_aggregates.append(sum(values) / len(values))
        
    return seed_aggregates

def run_analysis_pipeline(valid_results, config):
    """
    Executes the predefined analysis pipeline on the validated results.
    """
    analysis_results = {
        "comparisons": [],
        "bootstraps": []
    }
    
    metrics = ["cost", "fill_rate", "bullwhip"]
    
    # 1. Bootstraps for point estimates
    for algo in set(r["algorithm"] for r in valid_results):
        for scenario in set(r["scenario"] for r in valid_results):
            for metric in metrics:
                seed_data = extract_seed_level_data(valid_results, algo, scenario, metric)
                if len(seed_data) > 0:
                    ci = bootstrap_ci(seed_data, n_resamples=config.BOOTSTRAP_RESAMPLES, seed=42)
                    analysis_results["bootstraps"].append({
                        "algorithm": algo,
                        "scenario": scenario,
                        "metric": metric,
                        "bootstrap": ci
                    })

    # 2. Comparisons by family
    for family, pairs in config.RESEARCH_FAMILIES.items():
        family_results = []
        p_values = []
        
        for pair in pairs:
            algo1, algo2 = pair
            # For scenario robustness, the pair might be scenarios, not algos.
            # We differentiate based on the family name or content.
            is_scenario_comparison = family == "scenario_robustness"
            
            for metric in metrics:
                if is_scenario_comparison:
                    # Compare scenarios for MAPPO
                    scenario1, scenario2 = pair
                    group1 = extract_seed_level_data(valid_results, "mappo", scenario1, metric)
                    group2 = extract_seed_level_data(valid_results, "mappo", scenario2, metric)
                    name = f"mappo_{scenario1}_vs_{scenario2}_{metric}"
                else:
                    # Compare algos for in_distribution
                    group1 = extract_seed_level_data(valid_results, algo1, "in_distribution", metric)
                    group2 = extract_seed_level_data(valid_results, algo2, "in_distribution", metric)
                    name = f"{algo1}_vs_{algo2}_in_distribution_{metric}"
                    
                if len(group1) < 2 or len(group2) < 2:
                    continue
                    
                comp = compare_groups(group1, group2)
                d = cohens_d(group1, group2)
                
                family_results.append({
                    "comparison": name,
                    "metric": metric,
                    "test": comp,
                    "cohens_d": d,
                    "practically_significant": is_practically_significant(d, config.EFFECT_SIZE_THRESHOLD)
                })
                p_values.append(comp["p_value"])
                
        # 3. Apply FDR correction
        adj_p_values = benjamini_hochberg_correction(p_values)
        for i, res in enumerate(family_results):
            res["adjusted_p_value"] = adj_p_values[i]
            res["family"] = family
            analysis_results["comparisons"].append(res)
            
    return analysis_results

def generate_headline_claims(analysis_results):
    claims = []
    for comp in analysis_results["comparisons"]:
        d_text = f", Cohen's d: {comp['cohens_d']:.2f}" if comp["cohens_d"] is not None else ""
        claims.append(
            f"Comparison: {comp['comparison']} | "
            f"Test: {comp['test']['test_name']} | "
            f"p-value: {comp['test']['p_value']:.4f} | "
            f"adj p-value: {comp['adjusted_p_value']:.4f}{d_text} | "
            f"Sample size: n1={comp['test']['n1']}, n2={comp['test']['n2']}"
        )
    return claims
