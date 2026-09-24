
import numpy as np


def calculate_fill_rate(retailer_backlogs: list[int]) -> float:
    if not retailer_backlogs:
        return 0.0
    zero_backlog_count = sum(1 for b in retailer_backlogs if b == 0)
    return zero_backlog_count / len(retailer_backlogs)


def calculate_bullwhip(order_history: list[int], demand_history: list[int]) -> float:
    if len(order_history) <= 1 or len(demand_history) <= 1:
        return 1.0
    var_order = float(np.var(order_history, ddof=1))
    var_demand = float(np.var(demand_history, ddof=1))
    if var_demand == 0.0:
        return 1.0
    return var_order / var_demand
