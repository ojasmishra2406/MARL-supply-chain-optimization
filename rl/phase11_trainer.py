import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from rl.phase10_agent import Phase10Agent
from envs.supply_chain_env import SupplyChainParallelEnv
from forecasting.wrapper import DemandForecastWrapper
from forecasting.models import NaiveForecaster, MovingAverageForecaster, XGBoostForecaster
import supersuit as ss
import tempfile

class Phase11Trainer:
    def __init__(self, config_path, ablation_config=None, forecast_horizon=2, forecaster_type="ma"):
        self.ablation = ablation_config or {}
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.num_envs = self.config.get("num_envs", 8)
        self.centralized_critic = True
        self.shared_params = False
        
        # Temp sim config
        sim_config_path = self.config.get("simulator_config", "configs/phase1_simulator.yaml")
        with open(sim_config_path, "r") as f:
            sim_config = yaml.safe_load(f)
            
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, dir="configs") as temp_f:
            yaml.dump(sim_config, temp_f)
            temp_sim_config = temp_f.name
            
        base_env = SupplyChainParallelEnv(temp_sim_config, comm_enabled=False)
        self.agents_names = base_env.possible_agents
        self.num_agents = len(self.agents_names)
        
        # Forecasting setup
        if forecaster_type == "naive":
            self.forecaster = NaiveForecaster()
        elif forecaster_type == "xgboost":
            self.forecaster = XGBoostForecaster(window=5)
            # Need to fit the xgboost model or load it. 
            # In forecasting/models.py, it expects fit().
            # Assuming train_demand.csv is the standard distribution
            import pandas as pd
            df = pd.read_csv("data/forecasting/train_demand.csv")
            self.forecaster.fit(df["demand"].values)
        else:
            self.forecaster = MovingAverageForecaster(window=5)
            
        # Wrap environment with forecaster
        wrapped_env = DemandForecastWrapper(base_env, self.forecaster, forecast_horizon=forecast_horizon)
        
        vec_env = ss.pettingzoo_env_to_vec_env_v1(wrapped_env)
        self.env = ss.concat_vec_envs_v1(vec_env, self.num_envs)
        
        # Dynamically determine observation dimensions
        self.obs_dim = wrapped_env.observation_space("retailer").shape[0]
        self.global_obs_dim = self.obs_dim * self.num_agents
        
        print("Initializing agents...")
        self.device = torch.device("cpu") # Defaulting to CPU for safety during Phase 6 GPU occupation
        
        self.agents = {}
        for i, name in enumerate(self.agents_names):
            print(f"Initializing agent {name}...")
            self.agents[name] = Phase10Agent(
                obs_dim=self.obs_dim, 
                action_dim=wrapped_env.action_space(name).n, 
                num_nodes=self.num_agents,
                hidden_dim=64,
                agent_index=i,
                lr=self.ablation.get("learning_rate", float(self.config["learning_rate"]))
            ).to(self.device)
            
        print("Setting up optimizers...")
        self.optimizers = {
            a: self.agents[a].optimizer for a in self.agents_names
        }
        
        print("Resetting env...")
        obs, _ = self.env.reset(seed=42)
        print("Done init.")
        self.next_obs = obs
        
    def flatten_obs(self, obs_array):
        # The env is vectorized via supersuit, so obs_array is already shape (N*A, obs_dim)
        return torch.FloatTensor(obs_array).to(self.device)
        
    def _get_global_obs(self, obs_tensor):
        B_total = obs_tensor.shape[0]
        N = self.num_envs
        A = self.num_agents
        obs_reshaped = obs_tensor.view(N, A, -1)
        global_obs = obs_reshaped.view(N, A * self.obs_dim)
        global_obs_repeated = global_obs.repeat_interleave(A, dim=0)
        return global_obs_repeated
        
    def save(self, path):
        state_dicts = {a: self.agents[a].state_dict() for a in self.agents_names}
        torch.save(state_dicts, path)
        
    def load(self, path):
        state_dicts = torch.load(path, map_location=self.device)
        for a in self.agents_names:
            self.agents[a].load_state_dict(state_dicts[a])

    def train(self, total_updates=1):
        rollout_length = self.config.get("rollout_length", 128)
        N = self.num_envs
        A = self.num_agents
        
        buffers = {
            a: {
                "obs": torch.zeros((rollout_length, N, self.obs_dim), device=self.device),
                "global_obs": torch.zeros((rollout_length, N, self.global_obs_dim), device=self.device),
                "actions": torch.zeros((rollout_length, N), device=self.device),
                "logprobs": torch.zeros((rollout_length, N), device=self.device),
                "rewards": torch.zeros((rollout_length, N), device=self.device),
                "values": torch.zeros((rollout_length, N), device=self.device),
                "terminations": torch.zeros((rollout_length, N), device=self.device),
            } for a in self.agents_names
        }
        
        metrics_sum = {}
        
        for update in range(total_updates):
            total_env_reward = 0
            
            for step in range(rollout_length):
                obs_tensor = self.flatten_obs(self.next_obs)
                global_obs_tensor = self._get_global_obs(obs_tensor)
                
                actions_to_env = np.zeros(N * A, dtype=np.int64)
                
                with torch.no_grad():
                    for i, agent in enumerate(self.agents_names):
                        agent_obs = obs_tensor[i::A]
                        agent_gobs = global_obs_tensor[i::A]
                        
                        act, _, logprob, _, val = self.agents[agent].get_action_and_value(agent_obs, agent_gobs)
                        
                        buffers[agent]["obs"][step] = agent_obs
                        buffers[agent]["global_obs"][step] = agent_gobs
                        buffers[agent]["actions"][step] = act
                        buffers[agent]["logprobs"][step] = logprob
                        buffers[agent]["values"][step] = val.squeeze()
                            
                        actions_to_env[i::A] = act.cpu().numpy()

                step_action = actions_to_env
                next_obs, rews, terms, truncs, infos = self.env.step(step_action)
                
                scale = 100.0
                rews_scaled = torch.FloatTensor(rews).to(self.device) / scale
                terms_tensor = torch.FloatTensor(terms).to(self.device)
                
                total_env_reward += rews.sum()
                
                for i, agent in enumerate(self.agents_names):
                    buffers[agent]["rewards"][step] = rews_scaled[i::A]
                    buffers[agent]["terminations"][step] = terms_tensor[i::A]
                    
                self.next_obs = next_obs
            
            # --- PPO Update ---
            obs_tensor = self.flatten_obs(self.next_obs)
            global_obs_tensor = self._get_global_obs(obs_tensor)
            
            for i, agent in enumerate(self.agents_names):
                agent_obs = obs_tensor[i::A]
                agent_gobs = global_obs_tensor[i::A]
                with torch.no_grad():
                    next_value = self.agents[agent].get_value(agent_obs, agent_gobs).squeeze()
                    
                advantages = torch.zeros_like(buffers[agent]["rewards"])
                lastgaelam = 0
                for t in reversed(range(rollout_length)):
                    if t == rollout_length - 1:
                        nextnonterminal = 1.0 - terms_tensor[i::A]
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - buffers[agent]["terminations"][t + 1]
                        nextvalues = buffers[agent]["values"][t + 1]
                    delta = buffers[agent]["rewards"][t] + self.config["gamma"] * nextvalues * nextnonterminal - buffers[agent]["values"][t]
                    advantages[t] = lastgaelam = delta + self.config["gamma"] * self.config["gae_lambda"] * nextnonterminal * lastgaelam
                returns = advantages + buffers[agent]["values"]
                
                b_obs = buffers[agent]["obs"].reshape(-1, self.obs_dim)
                b_gobs = buffers[agent]["global_obs"].reshape(-1, self.global_obs_dim)
                b_actions = buffers[agent]["actions"].reshape(-1)
                b_logprobs = buffers[agent]["logprobs"].reshape(-1)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
                b_inds = np.arange(len(b_obs))
                
                opt = self.optimizers[agent]
                epochs = self.config.get("epochs", 4)
                minibatch_size = self.config.get("minibatch_size", 64)
                
                for epoch in range(epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, len(b_obs), minibatch_size):
                        end = start + minibatch_size
                        mb_inds = b_inds[start:end]
                        
                        _, _, newlogprob, entropy, newvalue = self.agents[agent].get_action_and_value(b_obs[mb_inds], b_gobs[mb_inds], b_actions[mb_inds])
                        
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()
                        
                        mb_advantages_mb = b_advantages[mb_inds]
                        pg_loss1 = -mb_advantages_mb * ratio
                        pg_loss2 = -mb_advantages_mb * torch.clamp(ratio, 1 - self.config["clip_epsilon"], 1 + self.config["clip_epsilon"])
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                        
                        v_loss = 0.5 * ((newvalue.squeeze() - b_returns[mb_inds]) ** 2).mean()
                        loss = pg_loss - self.config["entropy_coef"] * entropy.mean() + v_loss * self.config["value_coef"]
                        
                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.agents[agent].parameters(), self.config["max_grad_norm"])
                        opt.step()
                
                if agent == "retailer":
                    metrics_sum = {
                        "retailer_policy_loss": pg_loss.item(),
                        "retailer_value_loss": v_loss.item(),
                        "retailer_reward": total_env_reward / N
                    }
                    
        return metrics_sum
