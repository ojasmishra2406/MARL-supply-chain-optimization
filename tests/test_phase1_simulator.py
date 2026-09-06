import json
import os
import time

import hypothesis
import hypothesis.strategies as st
import numpy as np
import pytest

from simulator.core import SupplyChainSimulator


def get_config_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(repo_root, "configs", "phase1_simulator.yaml")


def test_four_echelon_simulator():
    sim = SupplyChainSimulator(get_config_path())
    assert sim.num_echelons == 4
    assert len(sim.state.echelons) == 4
    assert sim.capacity == 100


def test_inventory_conservation():
    """
    Mass balance:
    At any step, for an echelon:
    total_in_system = inventory + pipeline_inventory
    Initial total_in_system + cumulative_received = current total_in_system + cumulative_shipped
    Wait, cumulative_received is from the pipeline.
    More simply: for the entire simulator, the inventory is conserved if no orders are placed?
    Actually, let's track the exact items.
    For echelon i:
    inventory_{t} + pipeline_{t} + backlog_{t} - backlog_{t-1} = inventory_{t-1} + pipeline_{t-1} + received_by_pipeline_at_t - shipped_to_downstream_at_t
    
    Let's just run zero orders.
    If 0 orders are placed everywhere, pipeline receives 0.
    Retailer backlog will grow as demand comes in.
    """
    sim = SupplyChainSimulator(get_config_path(), seed=42)
    # Zero orders
    for _ in range(10):
        prev_state = sim._get_state_copy()
        state, info, _ = sim.step([0, 0, 0, 0])
        
        # Check Retailer
        # demand arrived = info["demand"][0]
        # shipped to customer = info["shipped"][0]
        # inventory changed by + arriving from pipeline - shipped
        # pipeline changed by + received from upstream - arriving
        
        for i in range(4):
            prev_e = prev_state.echelons[i]
            e = state.echelons[i]
            
            # Incoming to pipeline = info["shipped"][i+1] (for supplier, it's action[3]=0)
            incoming_to_pipeline = 0 if i == 3 else info["shipped"][i+1]
            shipped_out = info["shipped"][i]
            
            # Change in total system inventory for this echelon:
            # (new_inv + new_pipe) - (old_inv + old_pipe) == incoming_to_pipeline - shipped_out
            delta_system = (e.inventory + e.pipeline_inventory) - (prev_e.inventory + prev_e.pipeline_inventory)
            assert delta_system == incoming_to_pipeline - shipped_out


def test_capacity_clipping():
    sim = SupplyChainSimulator(get_config_path(), seed=10)
    # Give wholesaler 500 inventory
    sim.state.echelons[1].inventory = 500
    
    # Retailer requests 200 (which is > capacity 100)
    state, info, _ = sim.step([200, 0, 0, 0])
    
    # Wholesaler (index 1) should only ship 100 to retailer (index 0)
    # Shipped is at index 1 in the shipped array (which corresponds to what Echelon 1 shipped)
    assert info["shipped"][1] == 100
    
    # Check backlog: wholesaler received 200 demand, shipped 100, backlog = 100
    assert state.echelons[1].backlog == 100
    
    # Give distributor 50 inventory
    sim.state.echelons[2].inventory = 50
    # Wholesaler requests 80 (below capacity)
    state, info, _ = sim.step([0, 80, 0, 0])
    # Distributor should ship exactly 50 (limited by inventory, not capacity)
    assert info["shipped"][2] == 50
    assert state.echelons[2].backlog == 30


@hypothesis.given(st.lists(st.integers(min_value=1, max_value=10), min_size=4, max_size=4))
@hypothesis.settings(max_examples=50, deadline=None)
def test_lead_time_property(lt_config):
    sim = SupplyChainSimulator(get_config_path(), seed=42)
    sim.lead_times = lt_config
    sim.reset()
    
    # Give supplier an order of 50.
    L = lt_config[3]
    state, _, _ = sim.step([0, 0, 0, 50])
    
    for _ in range(L - 1):
        assert state.echelons[3].inventory == 0
        state, _, _ = sim.step([0, 0, 0, 0])
        
    # Before the arrival step, inventory is still 0
    assert state.echelons[3].inventory == 0
    # The arrival step (Step 1 + L)
    state, _, _ = sim.step([0, 0, 0, 0])
    
    assert state.echelons[3].inventory == 50


def test_seed_determinism():
    sim_a = SupplyChainSimulator(get_config_path(), seed=123)
    sim_b = SupplyChainSimulator(get_config_path(), seed=123)
    
    actions = [[10, 10, 10, 10] for _ in range(20)]
    
    traj_a = sim_a.serialize_trajectory(actions)
    traj_b = sim_b.serialize_trajectory(actions)
    
    assert traj_a == traj_b
    
    sim_c = SupplyChainSimulator(get_config_path(), seed=999)
    traj_c = sim_c.serialize_trajectory(actions)
    
    assert traj_a != traj_c


def test_demand_generation():
    sim = SupplyChainSimulator(get_config_path(), seed=42)
    state, info, _ = sim.step([0, 0, 0, 0])
    
    demand = info["demand"]
    assert demand[1] == 0
    assert demand[2] == 0
    assert demand[3] == 0
    assert demand[0] >= 0  # clipped at zero


def test_cost_accounting():
    sim = SupplyChainSimulator(get_config_path(), seed=42)
    sim.state.echelons[0].inventory = 10
    sim.state.echelons[0].backlog = 5
    
    # Cost holding=1, backlog=2, ordering=0.5
    # Retailer holding: 10 * 1 = 10
    # Retailer backlog: 5 * 2 = 10
    # Ordering: 20 * 0.5 = 10
    # Total for retailer = 30
    
    state, info, _ = sim.step([20, 0, 0, 0])
    
    # Wait, the step() modifies the state BEFORE cost calculation!
    # So the cost is based on ending inventory.
    # At start of step: inventory=10, backlog=5.
    # Demand is generated (e.g. 21).
    # Retailer requested = 21 + 5 = 26.
    # Retailer ships = min(26, 10, 100) = 10.
    # Retailer inventory becomes 0. Retailer backlog becomes 16.
    # Retailer places order 20. Cost ordering = 20 * 0.5 = 10.
    # Cost holding = 0 * 1 = 0.
    # Cost backlog = 16 * 2 = 32.
    # Total cost = 42.
    
    # Let's do a deterministic trace:
    # We can force demand by overriding the RNG just for a moment, or mocking.
    # Or we just check the output state.
    end_inv = state.echelons[0].inventory
    end_bl = state.echelons[0].backlog
    expected_cost = (end_inv * 1.0) + (end_bl * 2.0) + (20 * 0.5)
    
    # For echelons 1, 2, 3: inventory=0, backlog=order_received, ordering=0
    # Echelon 1 receives order 20. Backlog becomes 20. Cost = 20 * 2.0 = 40.
    for i in range(1, 4):
        expected_cost += (state.echelons[i].inventory * 1.0)
        expected_cost += (state.echelons[i].backlog * 2.0)
        expected_cost += (0 * 0.5)
        
    assert info["step_cost"] == expected_cost

def test_state_integrity():
    sim = SupplyChainSimulator(get_config_path(), seed=42)
    e = sim.state.echelons[0]
    assert hasattr(e, "inventory")
    assert hasattr(e, "backlog")
    assert hasattr(e, "pipeline_inventory")
    assert hasattr(e, "demand_history")
    assert hasattr(e, "last_order")

def test_invalid_input():
    sim = SupplyChainSimulator(get_config_path(), seed=42)
    with pytest.raises(ValueError):
        sim.step([0, 0, 0])  # Only 3 actions
    
    with pytest.raises(ValueError):
        sim.step([0, 0, -10, 0])  # Negative action

    # Test initialization validation
    import yaml
    with open(get_config_path(), "r") as f:
        config_data = yaml.safe_load(f)
        
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # 1. Not 4 echelons
    c1 = config_data.copy()
    c1["echelons"] = ["retailer"]
    p1 = os.path.join(repo_root, "configs", "test_invalid_1.yaml")
    with open(p1, "w") as f: yaml.dump(c1, f)
    with pytest.raises(ValueError, match="Expected exactly 4"): SupplyChainSimulator(p1)
    
    # 2. Mismatched lead times
    c2 = config_data.copy()
    c2["lead_time"] = [2, 2, 2]
    p2 = os.path.join(repo_root, "configs", "test_invalid_2.yaml")
    with open(p2, "w") as f: yaml.dump(c2, f)
    with pytest.raises(ValueError, match="Mismatch between echelons and lead_time"): SupplyChainSimulator(p2)
    
    # 3. Negative lead time
    c3 = config_data.copy()
    c3["lead_time"] = [2, -1, 3, 4]
    p3 = os.path.join(repo_root, "configs", "test_invalid_3.yaml")
    with open(p3, "w") as f: yaml.dump(c3, f)
    with pytest.raises(ValueError, match="Lead times cannot be negative"): SupplyChainSimulator(p3)

    # 4. Negative capacity
    c4 = config_data.copy()
    c4["capacity"] = {"units_per_step_per_echelon": -10}
    p4 = os.path.join(repo_root, "configs", "test_invalid_4.yaml")
    with open(p4, "w") as f: yaml.dump(c4, f)
    with pytest.raises(ValueError, match="Capacity cannot be negative"): SupplyChainSimulator(p4)
    
    # 5. Non-positive horizon
    c5 = config_data.copy()
    c5["horizon"] = 0
    p5 = os.path.join(repo_root, "configs", "test_invalid_5.yaml")
    with open(p5, "w") as f: yaml.dump(c5, f)
    with pytest.raises(ValueError, match="Horizon must be positive"): SupplyChainSimulator(p5)
    
    # 6. Negative cost
    c6 = config_data.copy()
    c6["cost"] = {"holding": -1.0, "backlog": 2.0, "ordering": 0.5}
    p6 = os.path.join(repo_root, "configs", "test_invalid_6.yaml")
    with open(p6, "w") as f: yaml.dump(c6, f)
    with pytest.raises(ValueError, match="Cost coefficients cannot be negative"): SupplyChainSimulator(p6)

    # Cleanup temp configs
    for p in [p1, p2, p3, p4, p5, p6]:
        if os.path.exists(p): os.remove(p)

def test_reset_with_seed():
    sim = SupplyChainSimulator(get_config_path(), seed=42)
    sim.step([0, 0, 0, 0])
    sim.reset(seed=999)
    assert sim._seed == 999

def test_no_rl_dependencies():
    import ast
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    core_path = os.path.join(repo_root, "simulator", "core.py")
    state_path = os.path.join(repo_root, "simulator", "state.py")
    
    banned_imports = {"gymnasium", "pettingzoo", "supersuit", "torch", "ray", "wandb", "streamlit"}
    
    for path in [core_path, state_path]:
        with open(path, "r") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split('.')[0] not in banned_imports
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert node.module.split('.')[0] not in banned_imports

def test_performance_benchmark():
    sim = SupplyChainSimulator(get_config_path(), seed=42)
    num_steps = 10000
    
    action = [10, 10, 10, 10]
    
    start_time = time.perf_counter()
    for _ in range(num_steps):
        # Prevent ending early due to horizon
        sim.current_step = 0
        sim.step(action)
    elapsed = time.perf_counter() - start_time
    
    steps_per_sec = num_steps / elapsed
    print(f"Performance: {steps_per_sec:.2f} steps/sec")
    
    assert steps_per_sec >= 10000, f"Performance too low: {steps_per_sec:.2f} steps/sec"

