import numpy as np

def benjamini_hochberg_correction(p_values):
    """
    Apply Benjamini-Hochberg FDR correction to a list of p-values.
    Returns a list of adjusted p-values in the original order.
    """
    if not p_values:
        return []
        
    p_values = np.array(p_values)
    n = len(p_values)
    
    # Sort indices
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    
    # Adjusted p-values
    adjusted_p_values = np.zeros(n)
    
    # BH logic: p_adj_i = min(p_i * n / rank_i, p_adj_{i+1})
    # Compute sequentially from highest rank to lowest rank to maintain monotonicity
    min_adj_p = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj_p = sorted_p_values[i] * n / rank
        min_adj_p = min(min_adj_p, adj_p)
        adjusted_p_values[sorted_indices[i]] = min_adj_p
        
    return adjusted_p_values.tolist()
