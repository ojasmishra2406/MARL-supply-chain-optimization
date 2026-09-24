import math

from simulator.core import SimulatorState


class OUTPolicy:
    def __init__(
        self,
        z_scores: list[float],
        lead_times: list[int],
        capacity: int,
        mu: float = 20.0,
        sigma: float = 5.0,
    ):
        self.z_scores = z_scores
        self.lead_times = lead_times
        self.capacity = capacity
        self.mu = mu
        self.sigma = sigma
        self.targets = []
        for z, L in zip(z_scores, lead_times):
            S = self.mu * (L + 1) + z * self.sigma * math.sqrt(L + 1)
            self.targets.append(S)

    def get_actions(self, state: SimulatorState) -> list[int]:
        actions = []
        for i, echelon in enumerate(state.echelons):
            ip = echelon.inventory + echelon.pipeline_inventory - echelon.backlog
            order = self.targets[i] - ip
            order = min(float(self.capacity), order)
            order = max(0.0, order)
            actions.append(round(order))
        return actions
