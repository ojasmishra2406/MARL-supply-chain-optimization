import numpy as np

def cohens_d(group1, group2):
    """
    Calculate Cohen's d for two groups.
    d = (mean1 - mean2) / pooled_std
    """
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return None
        
    var1 = np.var(group1, ddof=1) if n1 > 1 else 0
    var2 = np.var(group2, ddof=1) if n2 > 1 else 0
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
        
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return float(d)

def is_practically_significant(d, threshold=0.5):
    """
    Use |d| >= 0.5 as the predefined practical-significance threshold.
    """
    if d is None:
        return False
    return abs(d) >= threshold
