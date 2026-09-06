import dataclasses
from typing import List


@dataclasses.dataclass
class EchelonState:
    inventory: int = 0
    backlog: int = 0
    pipeline: List[int] = dataclasses.field(default_factory=list)
    pipeline_inventory: int = 0
    demand_history: List[int] = dataclasses.field(default_factory=list)
    last_order: int = 0

    def get_dict(self) -> dict:
        return {
            "inventory": self.inventory,
            "backlog": self.backlog,
            "pipeline": self.pipeline.copy(),
            "pipeline_inventory": self.pipeline_inventory,
            "demand_history": self.demand_history.copy(),
            "last_order": self.last_order,
        }


@dataclasses.dataclass
class SimulatorState:
    step: int
    echelons: List[EchelonState]
    total_cost: float

    def get_dict(self) -> dict:
        return {
            "step": self.step,
            "echelons": [e.get_dict() for e in self.echelons],
            "total_cost": self.total_cost,
        }
