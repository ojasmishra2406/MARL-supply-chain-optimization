import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminations: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation (GAE).
    rewards: (T, N)
    values: (T, N)
    terminations: (T, N)
    next_value: (N,)
    next_done: (N,)
    Returns:
        advantages: (T, N)
    """
    T, _N = rewards.shape
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - terminations[t + 1]
            nextvalues = values[t + 1]

        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam

    return advantages
