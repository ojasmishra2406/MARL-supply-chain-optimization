import numpy as np

def bootstrap_ci(data, stat_func=np.mean, n_resamples=1000, alpha=0.05, seed=42):
    """
    Bootstrap confidence interval generation.
    Must use minimum 1000 resamples.
    """
    if n_resamples < 1000:
        raise ValueError("Must use at least 1000 bootstrap resamples.")
        
    data = np.array(data)
    if len(data) == 0:
        return {"point_estimate": None, "lower_bound": None, "upper_bound": None, "resamples": n_resamples, "seed": seed}
        
    rng = np.random.default_rng(seed)
    
    point_estimate = float(stat_func(data))
    
    # Resample
    bootstrapped_stats = []
    n = len(data)
    
    # Generate all resamples at once for efficiency if data is not too large
    # but looping is fine for n_resamples=1000
    for _ in range(n_resamples):
        sample = rng.choice(data, size=n, replace=True)
        bootstrapped_stats.append(stat_func(sample))
        
    bootstrapped_stats = np.array(bootstrapped_stats)
    
    lower_bound = float(np.percentile(bootstrapped_stats, 100 * (alpha / 2)))
    upper_bound = float(np.percentile(bootstrapped_stats, 100 * (1 - alpha / 2)))
    
    return {
        "point_estimate": point_estimate,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "resamples": n_resamples,
        "seed": seed
    }
