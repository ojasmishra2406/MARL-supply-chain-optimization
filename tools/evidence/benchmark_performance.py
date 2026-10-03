"""
benchmark_performance.py — Forward-pass, environment step, GNN, and forecasting benchmarks.
"""
import os, sys, json, time, datetime, subprocess
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

FAILURES = []
WARMUP_PASSES = 10
BENCH_PASSES  = 100
N_ENV_STEPS   = 5000

def git_commit():
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"],
                                       cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"

def percentile(arr, p):
    return float(np.percentile(arr, p))

def bench_forward(model, dummy_input, passes=BENCH_PASSES, label=""):
    model.eval()
    with torch.no_grad():
        for _ in range(WARMUP_PASSES):
            model(dummy_input)
    times = []
    with torch.inference_mode():
        for _ in range(passes):
            t0 = time.perf_counter()
            model(dummy_input)
            times.append((time.perf_counter() - t0) * 1000)
    return {
        "label": label,
        "n_passes": passes,
        "mean_ms": round(float(np.mean(times)), 4),
        "median_ms": round(float(np.median(times)), 4),
        "p95_ms": round(percentile(times, 95), 4),
        "p99_ms": round(percentile(times, 99), 4),
        "min_ms": round(float(np.min(times)), 4),
        "max_ms": round(float(np.max(times)), 4),
        "passes_per_second": round(1000 / float(np.mean(times)), 1),
    }

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def bench_architectures():
    from rl.networks import ActorNetwork, CriticNetwork
    from rl.ppo import IPPOAgent, MAPPOAgent
    from rl.gnn_network import GNN_MAPPO_Actor, GNN_MAPPO_Critic

    results = {}
    device = torch.device("cpu")

    import yaml
    with open(os.path.join(ROOT, "configs/phase4_ippo.yaml")) as f:
        cfg = yaml.safe_load(f)
    hidden = cfg.get("hidden_size", 128)

    ippo_obs   = 5
    mappo_obs  = 5
    comm_obs   = 13
    action_dim = 11

    ippo = IPPOAgent(ippo_obs, action_dim, hidden).to(device)
    t_ippo, tr_ippo = count_parameters(ippo)
    dummy = torch.zeros(1, ippo_obs)
    fwd = bench_forward(ippo.actor, dummy, label="IPPO_actor")
    results["IPPO"] = {
        "class": type(ippo).__name__,
        "total_parameters": t_ippo,
        "trainable_parameters": tr_ippo,
        "obs_dim": ippo_obs,
        "action_dim": action_dim,
        "hidden_dim": hidden,
        "actor_layers": len([m for m in ippo.actor.modules() if isinstance(m, nn.Linear)]),
        "critic_layers": len([m for m in ippo.critic.modules() if isinstance(m, nn.Linear)]),
        "forward_pass": fwd,
    }

    global_obs = ippo_obs * 4
    mappo = MAPPOAgent(mappo_obs, global_obs, action_dim, hidden).to(device)
    t_mappo, tr_mappo = count_parameters(mappo)
    dummy_a = torch.zeros(1, mappo_obs)
    fwd_m = bench_forward(mappo.actor, dummy_a, label="MAPPO_actor")
    results["MAPPO"] = {
        "class": type(mappo).__name__,
        "total_parameters": t_mappo,
        "trainable_parameters": tr_mappo,
        "obs_dim": mappo_obs,
        "global_obs_dim": global_obs,
        "action_dim": action_dim,
        "hidden_dim": hidden,
        "forward_pass": fwd_m,
    }

    ippo_c = IPPOAgent(comm_obs, action_dim, hidden).to(device)
    t_ic, tr_ic = count_parameters(ippo_c)
    dummy_c = torch.zeros(1, comm_obs)
    fwd_ic = bench_forward(ippo_c.actor, dummy_c, label="IPPO_COMM_actor")
    results["IPPO_COMM"] = {
        "class": type(ippo_c).__name__,
        "obs_dim": comm_obs,
        "total_parameters": t_ic,
        "trainable_parameters": tr_ic,
        "forward_pass": fwd_ic,
    }

    comm_global = comm_obs * 4
    mappo_c = MAPPOAgent(comm_obs, comm_global, action_dim, hidden).to(device)
    t_mc, tr_mc = count_parameters(mappo_c)
    dummy_mc = torch.zeros(1, comm_obs)
    fwd_mc = bench_forward(mappo_c.actor, dummy_mc, label="MAPPO_COMM_actor")
    results["MAPPO_COMM"] = {
        "class": type(mappo_c).__name__,
        "obs_dim": comm_obs,
        "global_obs_dim": comm_global,
        "total_parameters": t_mc,
        "trainable_parameters": tr_mc,
        "forward_pass": fwd_mc,
    }

    num_nodes    = 4
    node_feat    = 5
    gnn_hidden   = 64
    gnn_actor  = GNN_MAPPO_Actor(num_nodes, node_feat, action_dim, gnn_hidden, agent_index=0).to(device)
    gnn_critic = GNN_MAPPO_Critic(num_nodes, node_feat, gnn_hidden).to(device)
    t_ga, tr_ga = count_parameters(gnn_actor)
    t_gc, tr_gc = count_parameters(gnn_critic)
    dummy_node = torch.zeros(1, node_feat)
    fwd_gnn = bench_forward(gnn_actor, dummy_node, label="GNN_MAPPO_actor")
    results["GNN_MAPPO"] = {
        "class_actor": type(gnn_actor).__name__,
        "class_critic": type(gnn_critic).__name__,
        "num_nodes": num_nodes,
        "node_feature_dim": node_feat,
        "gnn_hidden_dim": gnn_hidden,
        "actor_parameters": t_ga,
        "critic_parameters": t_gc,
        "total_gnn_parameters": t_ga + t_gc,
        "forward_pass_actor": fwd_gnn,
    }

    return results

def bench_environment(n_steps=N_ENV_STEPS):
    from envs.supply_chain_env import SupplyChainParallelEnv
    env = SupplyChainParallelEnv(os.path.join(ROOT, "configs/phase1_simulator.yaml"),
                                  comm_enabled=False)
    obs, _ = env.reset(seed=0)
    agents = env.agents

    reset_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        env.reset(seed=0)
        reset_times.append((time.perf_counter() - t0) * 1000)

    step_times = []
    obs, _ = env.reset(seed=0)
    done = False
    steps_done = 0
    while steps_done < n_steps:
        actions = {a: env.action_space(a).sample() for a in agents if a in obs}
        t0 = time.perf_counter()
        obs, _, terms, truncs, _ = env.step(actions)
        step_times.append((time.perf_counter() - t0) * 1000)
        steps_done += 1
        if all(terms.values()) or all(truncs.values()):
            obs, _ = env.reset(seed=0)

    total_s = sum(step_times) / 1000.0
    return {
        "n_steps_measured": steps_done,
        "total_elapsed_seconds": round(total_s, 3),
        "steps_per_second": round(steps_done / total_s, 1),
        "mean_step_ms": round(float(np.mean(step_times)), 4),
        "p95_step_ms": round(percentile(step_times, 95), 4),
        "reset_mean_ms": round(float(np.mean(reset_times)), 4),
        "source": "SupplyChainParallelEnv with random actions",
    }

def bench_forecasting():
    from forecasting.models import (NaiveForecaster, MovingAverageForecaster,
                                    ExponentialSmoothingForecaster,
                                    XGBoostForecaster, LSTMForecaster,
                                    evaluate_forecaster)
    import pandas as pd

    train_csv = os.path.join(ROOT, "data", "forecasting", "train_demand.csv")
    test_csv  = os.path.join(ROOT, "data", "forecasting", "test_demand_ood.csv")
    if not os.path.exists(train_csv):
        return {"error": f"Missing {train_csv}"}

    train_df = pd.read_csv(train_csv)
    train_series = train_df["demand"].values
    window = 5

    models = {
        "Naive":            NaiveForecaster(),
        "MovingAverage":    MovingAverageForecaster(window=window),
        "ExpSmoothing":     ExponentialSmoothingForecaster(alpha=0.3),
        "XGBoost":          XGBoostForecaster(window=window),
        "LSTM":             LSTMForecaster(window=window, epochs=20),
    }

    xgb_instance = models["XGBoost"]
    if not isinstance(xgb_instance, XGBoostForecaster):
        FAILURES.append(f"XGBOOST CLASS ASSERTION FAILED: got {type(xgb_instance).__name__}")

    results = {}
    for name, model in models.items():
        rec = {"class": type(model).__name__}
        if name in ("XGBoost", "LSTM"):
            t0 = time.perf_counter()
            model.fit(train_series)
            rec["fit_time_s"] = round(time.perf_counter() - t0, 3)

        history = list(train_series[:50])
        pred_times = []
        for _ in range(100):
            t0 = time.perf_counter()
            model.predict(history, horizon=1)
            pred_times.append((time.perf_counter() - t0) * 1000)
        rec["predict_mean_ms"] = round(float(np.mean(pred_times)), 4)
        rec["predict_p95_ms"]  = round(percentile(pred_times, 95), 4)

        metrics = evaluate_forecaster(model, train_df, window=window, horizon=1)
        rec["MAE"]   = round(metrics["MAE"], 4)
        rec["RMSE"]  = round(metrics["RMSE"], 4)
        rec["sMAPE"] = round(metrics["sMAPE"], 4)
        rec["Bias"]  = round(metrics["Bias"], 4)
        results[name] = rec

    if "XGBoost" in results and "class" in results["XGBoost"]:
        if results["XGBoost"]["class"] != "XGBoostForecaster":
            FAILURES.append(f"XGBoost reported class is {results['XGBoost']['class']}, not XGBoostForecaster")

    return results

def run():
    ts     = datetime.datetime.utcnow().isoformat() + "Z"
    commit = git_commit()
    arch = bench_architectures()
    env_bench = bench_environment()
    forecast_bench = bench_forecasting()

    result = {
        "model_architectures": arch,
        "environment": env_bench,
        "forecasting": forecast_bench,
        "failures": FAILURES,
        "timestamp": ts,
        "git_commit": commit,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    return result

if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2, default=str))
