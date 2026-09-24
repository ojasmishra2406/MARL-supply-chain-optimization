import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
sys.path.append(os.path.dirname(__file__))
from data_loader import load_registry, load_evaluations, load_statistical_analysis, generate_trajectory

st.set_page_config(page_title="MARL Supply Chain Dashboard", layout="wide")

# Load Data
registry = load_registry()
df_evals = load_evaluations()

st.sidebar.title("MARL Supply Chain")
page = st.sidebar.radio("Navigation", [
    "Overview", 
    "Model Registry",
    "Experiment Explorer",
    "Model Comparison", 
    "Scenario Analysis", 
    "Demand & Forecasting",
    "Live Simulation Replay",
    "Statistical Results",
    "Reproducibility"
])

if page == "Overview":
    st.title("Research Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    total_models = len(registry)
    evaluated_models = len([m for m in registry.values() if m.get("status") == "EVALUATED"])
    total_evals = len(df_evals)
    scenarios = df_evals["scenario"].nunique() if not df_evals.empty else 0
    
    col1.metric("Total Models", total_models)
    col2.metric("Evaluated Models", evaluated_models)
    col3.metric("Total Executed Scenarios", total_evals)
    col4.metric("Unique Scenarios", scenarios)
    
    st.subheader("Implementation Status")
    status_df = pd.DataFrame([{"Model ID": k, "Status": v["status"]} for k,v in registry.items()])
    if not status_df.empty:
        status_counts = status_df["Status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig = px.bar(status_counts, x="Status", y="Count", color="Status", title="Model Status Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No models found in registry.")

elif page == "Model Registry":
    st.title("Phase 13 Model Registry")
    st.markdown("Authoritative list of all models tracked by the reproducibility infrastructure.")
    
    if registry:
        reg_list = []
        for model_id, data in registry.items():
            meta = data.get("metadata", {})
            reg_list.append({
                "Model ID": model_id,
                "Algorithm": meta.get("algorithm", "unknown"),
                "Condition": meta.get("condition", "unknown"),
                "Seed": meta.get("seed", -1),
                "Status": data.get("status"),
                "Evaluations": len(data.get("evaluations", []))
            })
        st.dataframe(pd.DataFrame(reg_list), use_container_width=True)
    else:
        st.warning("Registry empty or missing.")

elif page == "Experiment Explorer":
    st.title("Experiment Explorer")
    st.markdown("Search and filter raw Phase 7 evaluation artifacts.")
    
    if not df_evals.empty:
        # Filters
        col1, col2, col3 = st.columns(3)
        algo_filter = col1.multiselect("Algorithm", df_evals["algorithm"].unique())
        scenario_filter = col2.multiselect("Scenario", df_evals["scenario"].unique())
        seed_filter = col3.multiselect("Evaluation Seed", df_evals["seed"].unique())
        
        filtered_df = df_evals.copy()
        if algo_filter:
            filtered_df = filtered_df[filtered_df["algorithm"].isin(algo_filter)]
        if scenario_filter:
            filtered_df = filtered_df[filtered_df["scenario"].isin(scenario_filter)]
        if seed_filter:
            filtered_df = filtered_df[filtered_df["seed"].isin(seed_filter)]
            
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.warning("No evaluation data found.")

elif page == "Model Comparison":
    st.title("Model Comparison")
    
    if not df_evals.empty:
        metric = st.selectbox("Select Metric", ["cost", "fill_rate", "bullwhip"])
        
        # We must aggregate over evaluation seeds to prevent pseudoreplication visually
        # Group by Model ID (which includes train seed) and scenario
        agg_df = df_evals.groupby(["model_id", "algorithm", "scenario"]).agg({metric: "mean"}).reset_index()
        
        fig = px.box(agg_df, x="algorithm", y=metric, color="algorithm", points="all",
                     title=f"Distribution of {metric} across Training Seeds (Aggregated over Scenarios)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("WARNING: Visual differences do not imply statistical significance. See 'Statistical Results' for Phase 8 FDR-adjusted p-values.")
    else:
        st.warning("No evaluation data available.")

elif page == "Scenario Analysis":
    st.title("Scenario Robustness Analysis")
    
    if not df_evals.empty:
        metric = st.selectbox("Select Metric", ["cost", "fill_rate", "bullwhip"])
        
        agg_df = df_evals.groupby(["algorithm", "scenario"]).agg({metric: "mean"}).reset_index()
        
        fig = px.bar(agg_df, x="scenario", y=metric, color="algorithm", barmode="group",
                     title=f"Mean {metric} by Scenario and Algorithm")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No evaluation data available.")

elif page == "Demand & Forecasting":
    st.title("Demand & Forecasting")
    st.markdown("Analyze demand generation and statistical forecasting implementations (Phase 9 & 11).")
    
    st.selectbox("Forecasting Model", ["Naive (Implemented)", "Moving Average (Implemented)", "Exponential Smoothing (Implemented)", "XGBoost (Phase 9)", "LSTM (Phase 9)"])
    st.selectbox("Data Regime", ["baseline", "high_variance"])
    st.info("Live trajectory overlay is available under 'Live Simulation Replay' for integrated GNN models.")

elif page == "Live Simulation Replay":
    st.title("Live Simulation Replay")
    st.markdown("Select an evaluated model to replay its decision trajectory in the environment.")
    
    if registry:
        valid_models = [m for m, data in registry.items() if data["status"] in ["TRAINED", "EVALUATED"]]
        selected_model = st.selectbox("Select Model", valid_models)
        selected_scenario = st.selectbox("Select Scenario", ["in_distribution", "demand_shift", "demand_spike"])
        selected_seed = st.number_input("Evaluation Seed", min_value=0, max_value=100, value=0)
        
        if st.button("Run Replay"):
            with st.spinner("Generating trajectory..."):
                try:
                    traj_df = generate_trajectory(selected_model, selected_scenario, selected_seed, registry)
                    st.success("Trajectory generated successfully (Mocked for dashboard structure demonstration).")
                    
                    st.subheader("Agent Inventory Over Time")
                    fig1 = px.line(traj_df, x="step", y="inventory", color="agent", title="Inventory Levels")
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    st.subheader("Agent Actions (Order Quantities)")
                    fig2 = px.line(traj_df, x="step", y="action", color="agent", title="Replenishment Orders")
                    st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to run simulation: {e}")
    else:
        st.warning("No trained models available in registry.")

elif page == "Statistical Results":
    st.title("Phase 8 Statistical Results")
    st.markdown("Authoritative statistical output from the Phase 8 analysis pipeline.")
    
    stats_md = load_statistical_analysis()
    st.markdown(stats_md)
    
elif page == "Reproducibility":
    st.title("Reproducibility Information")
    st.markdown("System information ensuring artifact traceability.")
    
    import platform
    import torch
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Environment")
        st.write(f"**Python:** {platform.python_version()}")
        st.write(f"**PyTorch:** {torch.__version__}")
        st.write(f"**OS:** {platform.system()} {platform.release()}")
    
    with col2:
        st.subheader("Hardware")
        st.write(f"**CUDA Available:** {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            st.write(f"**GPU:** {torch.cuda.get_device_name(0)}")
            
    st.markdown("---")
    st.markdown("To deterministically verify a specific model, run the CLI utility:")
    st.code("python tools/reproducibility/reproduce.py --model <MODEL_ID> --smoke-test", language="bash")
