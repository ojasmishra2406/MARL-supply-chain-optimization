import os
import sys
import platform
import torch
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Add project root to sys.path
WORKSPACE_ROOT = str(Path(__file__).resolve().parent.parent)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from dashboard.data_loader import (
    load_registry,
    load_registry_dataframe,
    load_evaluations,
    load_phase12_ablation_summary,
    load_phase12_scenario_breakdown,
    load_phase12_statistical_tests,
    load_forecasting_metrics,
    load_reproducibility_data,
    generate_trajectory
)

st.set_page_config(
    page_title="MARL Supply Chain Optimization Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load global cached data
registry_dict = load_registry()
df_registry = load_registry_dataframe()
df_evals = load_evaluations()

# Sidebar Navigation
st.sidebar.title("MARL Supply Chain")
st.sidebar.caption("Phases 1–14 Research & Operations Platform")

nav_sections = [
    "1. Overview",
    "2. Simulator & Replay",
    "3. Forecasting",
    "4. Agent Decisions",
    "5. Algorithm Comparison",
    "6. Scenario Analysis",
    "7. Statistics & Ablations",
    "8. Experiment Explorer",
    "9. Artifact Registry",
    "10. Reproducibility"
]

selected_page = st.sidebar.radio("Navigation", nav_sections)

st.sidebar.markdown("---")
st.sidebar.markdown("**System Health:** Operational")
st.sidebar.markdown(f"**Tracked Models:** {len(registry_dict)}")
st.sidebar.markdown(f"**Evaluation Runs:** {len(df_evals)}")

# ==============================================================================
# SECTION 1: OVERVIEW
# ==============================================================================
if selected_page == "1. Overview":
    st.title("Multi-Agent Reinforcement Learning for Supply Chain Optimization")
    st.markdown("### Executive Overview & Project Status")
    
    st.markdown("""
    This platform orchestrates multi-agent reinforcement learning (MARL), graph neural network (GNN) centralized critics,
    and supervised demand forecasting for multi-echelon inventory management under non-stationary stochastic demand shocks.
    """)
    
    # Top KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    total_models = len(df_registry)
    validated_models = len(df_registry[df_registry["research_status"] == "RESEARCH_VALIDATED"]) if not df_registry.empty else 0
    superseded_models = len(df_registry[df_registry["research_status"] == "HISTORICAL_SUPERSEDED"]) if not df_registry.empty else 0
    total_eval_episodes = len(df_evals)
    
    col1.metric("Registered Models", total_models, "84 on disk")
    col2.metric("Research Validated", validated_models, "Phase 6, 7, 9, 10, 11")
    col3.metric("Historical Superseded", superseded_models, "Phase 4, Phase 11 Hist")
    col4.metric("Logged Test Episodes", total_eval_episodes, "Across 6 Scenarios")
    
    st.markdown("---")
    
    st.subheader("Completed Phases & Architecture Roadmap")
    phases_data = [
        {"Phase": "Phase 1–3", "Focus": "Multi-Echelon Gym Environment & Simpy Engine", "Status": "COMPLETED", "Key Artifact": "envs/supply_chain.py"},
        {"Phase": "Phase 4", "Focus": "Single-Agent IPPO Baseline (50 updates)", "Status": "HISTORICAL_SUPERSEDED", "Key Artifact": "results/phase4_final_seed0.pt"},
        {"Phase": "Phase 5–7", "Focus": "MARL Scaling (IPPO, MAPPO, Comm, 5 seeds, 200 updates)", "Status": "RESEARCH_VALIDATED", "Key Artifact": "40 Production Checkpoints"},
        {"Phase": "Phase 8", "Focus": "Statistical Evaluation Harness & Welch/MWU Framework", "Status": "RESEARCH_VALIDATED", "Key Artifact": "PHASE8_STATISTICAL_ANALYSIS.md"},
        {"Phase": "Phase 9", "Focus": "Demand Forecasting Engine (Naive, MA, ExpSmooth, XGBoost)", "Status": "RESEARCH_VALIDATED", "Key Artifact": "results/phase9/forecasting_metrics.csv"},
        {"Phase": "Phase 10", "Focus": "Graph Neural Network Critic Topology (GNN-MAPPO)", "Status": "RESEARCH_VALIDATED", "Key Artifact": "10 GNN Checkpoints"},
        {"Phase": "Phase 11", "Focus": "Controlled Forecasting Replication (Independent Seeding)", "Status": "RESEARCH_VALIDATED", "Key Artifact": "results/phase11_controlled/"},
        {"Phase": "Phase 12", "Focus": "Comprehensive 56-Model Ablations & Statistical Validation", "Status": "CONDITIONAL_PASS", "Key Artifact": "results/phase12/ablation_statistical_tests.json"},
        {"Phase": "Phase 13", "Focus": "Dual-Hash Checkpoint Integrity & Deterministic Replay", "Status": "PASS", "Key Artifact": "models/registry.json"},
        {"Phase": "Phase 14", "Focus": "Interactive Multi-Echelon Analytics Dashboard", "Status": "COMPLETED", "Key Artifact": "dashboard/app.py"}
    ]
    st.dataframe(pd.DataFrame(phases_data), use_container_width=True)
    
    st.markdown("---")
    st.subheader("Critical Scientific Caveats")
    st.warning(r"""
    **Methodological Boundaries & Statistical Cautions:**
    1. **Decentralization Advantage:** Independent PPO (IPPO) significantly suppresses the Bullwhip Effect compared to centralized MAPPO ($p = 0.0092$), but under False Discovery Rate (FDR) control across Family D, $q = 0.0733$.
    2. **Sample Size & Power Constraints ($N=5$):** Non-significant differences (e.g. GNN vs. MLP cost, $p = 0.3401$; Communication cost, $p = 0.3337$) **cannot be interpreted as proof of equivalence**. Power to detect small-to-medium effects at $N=5$ is $< 20\%$.
    3. **Training Budget Asymmetry:** Forecasting ablations in Family B were trained for 50 updates versus 200 updates for architectural baselines, confounding cross-family comparisons.
    4. **Offline vs. Online Forecasting Gap:** XGBoost achieves lowest offline forecast error (MAE 1.28), but negative out-of-distribution bias ($-5.27$) causes higher stockout penalties in online control.
    """)

# ==============================================================================
# SECTION 2: SIMULATOR & REPLAY
# ==============================================================================
elif selected_page == "2. Simulator & Replay":
    st.title("Simulator Configuration & Policy Replay")
    st.markdown("Inspect multi-echelon simulation environment parameters and replay evaluated model trajectories.")
    
    tab_cfg, tab_replay = st.tabs(["Simulation Configurations", "Deterministic Trajectory Replay"])
    
    with tab_cfg:
        st.subheader("Echelon Parameters & Penalty Structures")
        config_options = {
            "Baseline Simulator (Standard Variance)": "configs/phase1_simulator.yaml",
            "Out-of-Distribution: Capacity Disruption": "configs/eval_scenarios/capacity_disruption.yaml",
            "Out-of-Distribution: Demand Spike": "configs/eval_scenarios/demand_spike.yaml",
            "Out-of-Distribution: In-Distribution Scenario": "configs/eval_scenarios/in_distribution.yaml",
            "Initial IPPO Configuration": "configs/phase4_ippo.yaml",
            "Primary Baselines Experiment Matrix": "configs/phase12/primary_baselines.yaml"
        }
        chosen_cfg = st.selectbox("Select Simulation Configuration", list(config_options.keys()))
        cfg_file = config_options[chosen_cfg]
        full_cfg_path = os.path.join(WORKSPACE_ROOT, cfg_file)
        
        if os.path.exists(full_cfg_path):
            with open(full_cfg_path, "r") as f:
                content = f.read()
            st.code(content, language="yaml")
        else:
            st.error(f"Configuration file not found: {cfg_file}")
            
    with tab_replay:
        st.subheader("Deterministic Episode Replay")
        st.markdown("Replay a 52-step simulation episode for an evaluated model checkpoint without running training.")
        
        if not df_registry.empty:
            valid_models = df_registry[df_registry["research_status"] == "RESEARCH_VALIDATED"]["model_id"].tolist()
            if not valid_models:
                valid_models = df_registry["model_id"].tolist()
            if valid_models:
                col_m, col_s, col_seed = st.columns(3)
                default_idx = valid_models.index("phase6_ippo_baseline_s0") if "phase6_ippo_baseline_s0" in valid_models else 0
                sel_model = col_m.selectbox("Select Model", valid_models, index=default_idx)
                sel_sc = col_s.selectbox("Scenario", ["in_distribution", "demand_shift", "lead_time_shift", "capacity_disruption", "demand_spike", "combined_shift"])
                sel_seed = col_seed.number_input("Evaluation Seed", min_value=0, max_value=100, value=0)
                
                if st.button("Run Replay Simulation"):
                    with st.spinner("Executing 52-step simulation rollout..."):
                        traj_df = generate_trajectory(sel_model, sel_sc, sel_seed, registry_dict)
                        
                        if "error" in traj_df.columns:
                            st.error(traj_df.iloc[0]["error"])
                        else:
                            st.success(f"Successfully replayed 52 steps for model `{sel_model}` under scenario `{sel_sc}` (seed {sel_seed}).")
                            
                            fig_inv = px.line(traj_df, x="step", y="inventory", color="agent", title="Echelon Inventory Levels Over 52 Weeks")
                            st.plotly_chart(fig_inv, use_container_width=True)
                            
                            fig_act = px.line(traj_df, x="step", y="action", color="agent", title="Replenishment Orders (Actions) Over 52 Weeks")
                            st.plotly_chart(fig_act, use_container_width=True)
                            
                            fig_back = px.line(traj_df, x="step", y="backlog", color="agent", title="Customer Backlog Spikes Over Time")
                            st.plotly_chart(fig_back, use_container_width=True)
            else:
                st.info("No validated checkpoints available for replay.")
        else:
            st.warning("Model registry is empty.")

# ==============================================================================
# SECTION 3: FORECASTING
# ==============================================================================
elif selected_page == "3. Forecasting":
    st.title("Demand Forecasting Benchmark (Phase 9 & Phase 11)")
    st.markdown("Compare supervised and heuristic demand prediction models and analyze why offline predictive accuracy does not translate to online MARL policy gains.")
    
    df_forecasting = load_forecasting_metrics()
    
    if not df_forecasting.empty:
        st.subheader("Offline Forecasting Accuracy Metrics")
        st.dataframe(df_forecasting, use_container_width=True)
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            fig_mae = px.bar(df_forecasting, x="Model", y="MAE", color="Dataset", barmode="group",
                             title="Mean Absolute Error (MAE) [Lower is Better]")
            st.plotly_chart(fig_mae, use_container_width=True)
        with col_f2:
            fig_bias = px.bar(df_forecasting, x="Model", y="Bias", color="Dataset", barmode="group",
                              title="Forecasting Bias (Under vs Overforecasting)")
            st.plotly_chart(fig_bias, use_container_width=True)
            
        st.markdown("---")
        st.subheader("Offline Predictive Accuracy vs. Online Policy Translation")
        st.markdown(r"""
        **The Supervised-to-RL Translation Dilemma:**
        - **Supervised Benchmark:** XGBoost dominates in-distribution (MAE $1.28$ vs. $3.47$ for Moving Average) and out-of-distribution (MAE $8.48$ vs. $9.37$).
        - **Online RL Rollouts:** When GNN-MAPPO is conditioned on XGBoost predictions in real-time, global cost shows no significant change ($p = 0.7157$), customer service level drops ($-5.73\%$, $p = 0.2086$), and Bullwhip ratio rises ($+46.82\%$, raw $p = 0.0840$).
        - **Causal Root Cause:** Under distribution shifts, XGBoost displays severe **negative forecasting bias ($-5.27$ units)**. In inventory management, under-forecasting is severely penalized by stockout backlogs, making a slightly biased complex model worse than a simple zero-bias Moving Average filter.
        """)
    else:
        st.warning("Forecasting metrics artifact `results/phase9/forecasting_metrics.csv` unavailable.")

# ==============================================================================
# SECTION 4: AGENT DECISIONS
# ==============================================================================
elif selected_page == "4. Agent Decisions":
    st.title("Multi-Echelon Agent Decision Dynamics")
    st.markdown("Analyze policy ordering behavior, echelon bullwhip transmission, and buffer management across Retailer, Wholesaler, Distributor, and Manufacturer.")
    
    if not df_evals.empty:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            fig_bw = px.box(df_evals, x="algorithm", y="bullwhip", color="algorithm", points="outliers",
                            title="Bullwhip Ratio Distribution Across Algorithms")
            st.plotly_chart(fig_bw, use_container_width=True)
        with col_m2:
            fig_fr = px.box(df_evals, x="algorithm", y="fill_rate", color="algorithm", points="outliers",
                            title="Customer Fill Rate Distribution Across Algorithms")
            st.plotly_chart(fig_fr, use_container_width=True)
            
        st.markdown("### The Bullwhip Transmission Mechanism")
        st.markdown(r"""
        - **Independent PPO (IPPO):** Operates with narrow, stable replenishment orders ($1.48 \pm 0.47$ bullwhip ratio). Each agent optimizes local backlog penalties without coordinating safety buffers.
        - **Centralized MAPPO:** Centralized critics share value gradients across echelons, causing upstream agents (Distributor, Manufacturer) to build large safety stocks to shield downstream fill rate. This creates large upstream order oscillations ($6.33 \pm 2.37$ bullwhip ratio, $+329\%$ higher than IPPO).
        """)
    else:
        st.warning("No evaluation episode data found.")

# ==============================================================================
# SECTION 5: ALGORITHM COMPARISON
# ==============================================================================
elif selected_page == "5. Algorithm Comparison":
    st.title("Controlled Algorithm Benchmark")
    st.markdown("Compare MARL architectures (IPPO, IPPO_COMM, MAPPO, MAPPO_COMM, GNN-MAPPO) under identical training budgets (200 updates, $N=5$ seeds).")
    
    df_summary = load_phase12_ablation_summary()
    
    if not df_summary.empty:
        regime_choice = st.radio("Select Training Regime", ["baseline", "high_variance"], horizontal=True)
        filtered_sum = df_summary[df_summary["regime"] == regime_choice]
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_cost = px.bar(filtered_sum.groupby("algorithm")["cost"].mean().reset_index(),
                              x="algorithm", y="cost", color="algorithm",
                              title=f"Mean Operational Cost ({regime_choice.title()} Regime)")
            st.plotly_chart(fig_cost, use_container_width=True)
        with col_c2:
            fig_eff = px.bar(filtered_sum.groupby("algorithm")["cost_per_sl"].mean().reset_index(),
                             x="algorithm", y="cost_per_sl", color="algorithm",
                             title=f"Cost / Service Level Ratio [Lower is Better] ({regime_choice.title()} Regime)")
            st.plotly_chart(fig_eff, use_container_width=True)
            
        st.subheader("Model-Level Metric Table (Averaged Over 30 Evaluation Episodes)")
        st.dataframe(filtered_sum[["algorithm", "training_seed", "cost", "service_level", "bullwhip", "cost_per_sl"]], use_container_width=True)
    else:
        st.warning("Authoritative Phase 12 summary data missing.")

# ==============================================================================
# SECTION 6: SCENARIO ANALYSIS
# ==============================================================================
elif selected_page == "6. Scenario Analysis":
    st.title("Out-of-Distribution Scenario Robustness")
    st.markdown("Analyze policy resilience across 6 distribution shifts: In-Distribution, Demand Shift, Lead Time Shift, Capacity Disruption, Demand Spike, and Combined Shift.")
    
    df_sc = load_phase12_scenario_breakdown()
    
    if not df_sc.empty:
        metric_choice = st.selectbox("Select Performance Metric", ["cost", "service_level", "bullwhip"])
        mean_col = f"{metric_choice}_mean"
        
        fig_sc = px.bar(df_sc[df_sc["regime"] == "baseline"], x="scenario", y=mean_col, color="algorithm", barmode="group",
                        title=f"Mean {metric_choice.title()} Across Distribution Shifts (Baseline Regime)")
        st.plotly_chart(fig_sc, use_container_width=True)
        
        st.subheader("Scenario Breakdown Data Table")
        st.dataframe(df_sc, use_container_width=True)
    else:
        st.warning("Scenario breakdown data unavailable.")

# ==============================================================================
# SECTION 7: STATISTICS & ABLATIONS
# ==============================================================================
elif selected_page == "7. Statistics & Ablations":
    st.title("Authoritative Phase 12 Statistical Ablations")
    st.markdown("Full hypothesis testing matrix recomputed directly from 1,680 raw evaluation episodes across 56 models.")
    
    df_stats, raw_stats_json = load_phase12_statistical_tests()
    
    if not df_stats.empty:
        families = df_stats["Family"].unique().tolist()
        sel_fam = st.multiselect("Filter by Hypothesis Family", families, default=families)
        
        filtered_stats = df_stats[df_stats["Family"].isin(sel_fam)]
        
        # Display key summary KPI badges
        sig_nom = len(filtered_stats[filtered_stats["Nominal Sig (p < 0.05)"]])
        sig_fdr = len(filtered_stats[filtered_stats["FDR Sig (q < 0.05)"]])
        
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Total Hypothesis Tests", len(filtered_stats))
        col_s2.metric("Nominal Significant (p < 0.05)", sig_nom, "Unadjusted Welch")
        col_s3.metric("FDR Significant (q < 0.05)", sig_fdr, "BH-FDR Corrected")
        
        st.dataframe(filtered_stats[[
            "Family", "Comparison", "Metric", "nx", "ny",
            "Mean A", "SD A", "Mean B", "SD B",
            "Welch t", "df", "Raw p-value", "MWU p-value",
            "Hedges g", "FDR q-value", "Nominal Sig (p < 0.05)", "FDR Sig (q < 0.05)"
        ]], use_container_width=True)
        
        st.info(r"""
        **Statistical Note on FDR vs. Nominal Significance:**
        - Two tests achieve nominal significance at unadjusted $\alpha = 0.05$: **MAPPO vs. IPPO Baseline Bullwhip ($p = 0.0092$)** and **MAPPO vs. IPPO Baseline Cost/SL ($p = 0.0380$)**.
        - However, after Benjamini-Hochberg FDR control within Family D (8 tests), their adjusted $q$-values are **$q = 0.0733$** and **$q = 0.0972$**. Neither survives strict FDR control at $q < 0.05$.
        - All GNN topology and communication cost differences are statistically non-significant ($p > 0.30$).
        """)
    else:
        st.warning("Statistical test results missing.")

# ==============================================================================
# SECTION 8: EXPERIMENT EXPLORER
# ==============================================================================
elif selected_page == "8. Experiment Explorer":
    st.title("Experiment Run Explorer")
    st.markdown("Filter, search, and export raw evaluation records.")
    
    if not df_evals.empty:
        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        sel_p = col_e1.multiselect("Phase", df_evals["phase"].unique(), default=df_evals["phase"].unique())
        sel_a = col_e2.multiselect("Algorithm", df_evals["algorithm"].unique(), default=df_evals["algorithm"].unique())
        sel_sc = col_e3.multiselect("Scenario", df_evals["scenario"].unique(), default=df_evals["scenario"].unique())
        sel_c = col_e4.selectbox("Controlled Replication Only?", ["All", "Controlled Only", "Historical Only"])
        
        df_filtered = df_evals[
            (df_evals["phase"].isin(sel_p)) &
            (df_evals["algorithm"].isin(sel_a)) &
            (df_evals["scenario"].isin(sel_sc))
        ]
        if sel_c == "Controlled Only":
            df_filtered = df_filtered[df_filtered["is_controlled"] == True]
        elif sel_c == "Historical Only":
            df_filtered = df_filtered[df_filtered["is_controlled"] == False]
            
        st.markdown(f"**Showing {len(df_filtered)} matching evaluation episodes:**")
        st.dataframe(df_filtered, use_container_width=True)
        
        csv_data = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button("Export Filtered Episodes as CSV", csv_data, "filtered_episodes.csv", "text/csv")
    else:
        st.warning("No evaluation records available.")

# ==============================================================================
# SECTION 9: ARTIFACT REGISTRY
# ==============================================================================
elif selected_page == "9. Artifact Registry":
    st.title("Phase 13 Model & Artifact Registry")
    st.markdown("Authoritative row-level inventory of all 84 models with orthogonal status separation.")
    
    if not df_registry.empty:
        col_r1, col_r2, col_r3 = st.columns(3)
        int_stat = col_r1.multiselect("Integrity Status", df_registry["integrity_status"].unique(), default=df_registry["integrity_status"].unique())
        eval_stat = col_r2.multiselect("Evaluation Status", df_registry["evaluation_status"].unique(), default=df_registry["evaluation_status"].unique())
        res_stat = col_r3.multiselect("Research Status", df_registry["research_status"].unique(), default=df_registry["research_status"].unique())
        
        filtered_reg = df_registry[
            (df_registry["integrity_status"].isin(int_stat)) &
            (df_registry["evaluation_status"].isin(eval_stat)) &
            (df_registry["research_status"].isin(res_stat))
        ]
        
        st.dataframe(filtered_reg[[
            "model_id", "phase", "algorithm", "training_updates",
            "integrity_status", "evaluation_status", "research_status", "status",
            "canonical_sha256", "provenance_notes"
        ]], use_container_width=True)
        
        st.markdown("""
        **Orthogonal Status Contract:**
        - **Integrity Status (`ARTIFACT_VALID`):** Verifies that the checkpoint byte stream or state-dict matches its recorded SHA-256 hash on disk.
        - **Research Status (`RESEARCH_VALIDATED` vs `HISTORICAL_SUPERSEDED` vs `TRAINING_INCOMPLETE`):** Validates scientific acceptance. Valid hashes do not imply research validation (e.g., Phase 4 is `HISTORICAL_SUPERSEDED`; Phase 12 diagnostic is `TRAINING_INCOMPLETE`).
        """)
    else:
        st.warning("Registry empty or missing.")

# ==============================================================================
# SECTION 10: REPRODUCIBILITY
# ==============================================================================
elif selected_page == "10. Reproducibility":
    st.title("Provenance, Verification & Reproducibility")
    st.markdown("Cryptographic provenance, environment metadata, and automated audit replay results.")
    
    rep_data, int_data = load_reproducibility_data()
    
    col_env1, col_env2 = st.columns(2)
    with col_env1:
        st.subheader("Host Environment")
        st.write(f"**Python Runtime:** {platform.python_version()}")
        st.write(f"**PyTorch Version:** {torch.__version__}")
        st.write(f"**Operating System:** {platform.system()} {platform.release()}")
    with col_env2:
        st.subheader("Hardware & Acceleration")
        st.write(f"**CUDA Detected:** {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            st.write(f"**Device Name:** {torch.cuda.get_device_name(0)}")
            
    st.markdown("---")
    st.subheader("Deterministic Evaluation Replay Audit")
    if rep_data:
        st.success(f"Audit Status: {rep_data.get('status', 'PASSED')}")
        st.json(rep_data)
    else:
        st.info("Automated reproducibility report `reproducibility_verification.json` not loaded.")
        
    st.markdown("---")
    st.subheader("Dual-Hash Checkpoint Integrity Summary")
    if int_data:
        st.json(int_data.get("summary", int_data))
    else:
        st.info("Checkpoint integrity report `checkpoint_integrity_report.json` not loaded.")
        
    st.markdown("---")
    st.subheader("Reproducibility CLI Usage")
    st.code("""
# Run the automated reproducibility audit:
python tools/reproducibility/check_reproducibility.py

# Run targeted dual-hash integrity tests:
pytest tests/test_targeted_dual_hash.py -v

# Run status separation tests:
pytest tests/test_status_separation.py -v
    """, language="bash")
