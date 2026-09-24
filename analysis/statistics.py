import numpy as np
import scipy.stats as stats

def check_normality(data):
    """
    Shapiro-Wilk normality test.
    Returns statistic, p-value, and sample size.
    """
    if len(data) < 3:
        return {"statistic": None, "p_value": None, "n": len(data), "is_normal": False}
        
    stat, p = stats.shapiro(data)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "n": len(data),
        "is_normal": p >= 0.05
    }

def compare_groups(group1, group2):
    """
    Selects and runs the appropriate comparison test (Welch's t-test or Mann-Whitney U)
    based on normality of the distributions.
    """
    norm1 = check_normality(group1)
    norm2 = check_normality(group2)
    
    # If both groups are sufficiently compatible with parametric assumptions, use Welch's t-test.
    # We define "compatible" as both passing Shapiro-Wilk (p >= 0.05).
    # Since small samples might fail to reject normality incorrectly, Welch's t-test is generally robust,
    # but the instruction says: "when the comparison is sufficiently compatible with parametric assumptions. 
    # Otherwise use Mann-Whitney U test."
    
    use_parametric = norm1["is_normal"] and norm2["is_normal"]
    
    if use_parametric:
        stat, p = stats.ttest_ind(group1, group2, equal_var=False) # Welch's t-test
        test_name = "Welch's t-test"
    else:
        stat, p = stats.mannwhitneyu(group1, group2, alternative="two-sided")
        test_name = "Mann-Whitney U test"
        
    return {
        "test_name": test_name,
        "statistic": float(stat),
        "p_value": float(p),
        "n1": len(group1),
        "n2": len(group2)
    }
