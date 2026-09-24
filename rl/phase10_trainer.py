import torch
import numpy as np
import supersuit as ss
import yaml
import torch.nn as nn
from rl.phase10_agent import Phase10Agent
from envs.supply_chain_env import SupplyChainParallelEnv
import time

class Phase10Trainer:
    def __init__(self, config_path, ablation_config):
        self.ablation = ablation_config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.shared_params = self.ablation.get("parameter_sharing", False)
        self.centralized_critic = self.ablation.get("centralized_critic", True)
        self.comm_enabled = self.ablation.get("communication", False)
        self.seed = self.ablation.get("seed", 0)
        
        with open(self.config["simulator_config"], "r") as f:
            sim_cfg = yaml.safe_load(f)
            
        if "reward_weights" in self.ablation:
            sim_cfg["cost"] = self.ablation["reward_weights"]
            
        import os
        temp_sim_config = f"configs/temp_phase6_sim_{self.seed}_{os.getpid()}.yaml"
        with open(temp_sim_config, "w") as f:
            yaml.dump(sim_cfg, f)
            
        self.num_envs = self.config.get("num_envs", 8)
        
        # SuperSuit Vectorization
        base_env = SupplyChainParallelEnv(temp_sim_config, comm_enabled=self.comm_enabled)
        self.agents_names = base_env.possible_agents
        self.num_agents = len(self.agents_names)
        
        vec_env = ss.pettingzoo_env_to_vec_env_v1(base_env)
        self.env = ss.concat_vec_envs_v1(vec_env, self.num_envs)
        
        self.obs_dim = 13 if self.comm_enabled else 5
        self.global_obs_dim = self.obs_dim * self.num_agents if self.centralized_critic else self.obs_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.agents = {}
        if self.shared_params:
            agent = Phase10Agent(
                obs_dim=self.obs_dim,
                action_dim=base_env.action_space(self.agents_names[0]).n,
                num_nodes=self.num_agents,
                hidden_dim=64,
                agent_index=0,
                lr=self.ablation.get("learning_rate", float(self.config["learning_rate"]))
            ).to(self.device)
            for name in self.agents_names:
                self.agents[name] = agent
        else:
            for i, name in enumerate(self.agents_names):
                self.agents[name] = Phase10Agent(
                    obs_dim=self.obs_dim,
                    action_dim=base_env.action_space(name).n,
                    num_nodes=self.num_agents,
                    hidden_dim=64,
                    agent_index=i,
                    lr=self.ablation.get("learning_rate", float(self.config["learning_rate"]))
                ).to(self.device)
                
        self.lr = self.ablation.get("learning_rate", float(self.config["learning_rate"]))
        self.optimizers = {
            a: torch.optim.Adam(self.agents[a].parameters(), lr=self.lr, eps=1e-5)
            for a in (self.agents_names if not self.shared_params else [self.agents_names[0]])
        }
        
        obs, _ = self.env.reset(seed=self.seed)
        self.next_obs = obs

    def flatten_obs(self, obs_batch):
        if not self.comm_enabled:
            return torch.FloatTensor(obs_batch).to(self.device)
        
        B = obs_batch['state'].shape[0]
        flat = torch.zeros((B, 13), device=self.device)
        flat[:, 0:5] = torch.FloatTensor(obs_batch['state']).to(self.device)
        flat[:, 5:9] = torch.FloatTensor(obs_batch['message_upstream']).to(self.device)
        flat[:, 9:13] = torch.FloatTensor(obs_batch['message_downstream']).to(self.device)
        return flat

    def _get_global_obs(self, obs_tensor):
        # obs_tensor is shape (num_envs * num_agents, obs_dim)
        # We need (num_envs, num_agents * obs_dim) then repeat it for each agent?
        # Actually, if we reshape to (num_envs, num_agents, obs_dim), we can flatten the last two.
        B_total = obs_tensor.shape[0]
        N = self.num_envs
        A = self.num_agents
        obs_reshaped = obs_tensor.view(N, A, -1)
        global_obs = obs_reshaped.view(N, A * self.obs_dim)
        # Repeat for each agent so shape matches (B_total, global_obs_dim)
        global_obs_repeated = global_obs.repeat_interleave(A, dim=0)
        return global_obs_repeated

    def train(self, total_updates=1, use_wandb=False):
        rollout_length = self.config["rollout_length"]
        N = self.num_envs
        A = self.num_agents
        
        buffers = {
            a: {
                "obs": torch.zeros((rollout_length, N, self.obs_dim), device=self.device),
                "global_obs": torch.zeros((rollout_length, N, self.global_obs_dim), device=self.device),
                "actions": torch.zeros((rollout_length, N), dtype=torch.long, device=self.device),
                "logprobs": torch.zeros((rollout_length, N), device=self.device),
                "rewards": torch.zeros((rollout_length, N), device=self.device),
                "values": torch.zeros((rollout_length, N), device=self.device),
                "terminations": torch.zeros((rollout_length, N), device=self.device),
            } for a in self.agents_names
        }
        
        if self.comm_enabled:
            for a in self.agents_names:
                buffers[a]["comm_actions"] = torch.zeros((rollout_length, N, 4), device=self.device)

        metrics_sum = {}
        
        for update in range(total_updates):
            total_env_reward = 0
            
            for step in range(rollout_length):
                obs_tensor = self.flatten_obs(self.next_obs)
                global_obs_tensor = self._get_global_obs(obs_tensor)
                
                actions_to_env = np.zeros(N * A, dtype=np.int64)
                if self.comm_enabled:
                    comm_actions_to_env = np.zeros((N * A, 4), dtype=np.float32)
                
                with torch.no_grad():
                    for i, agent in enumerate(self.agents_names):
                        # Slicing: start at `i`, step by `A`
                        agent_obs = obs_tensor[i::A]
                        agent_gobs = global_obs_tensor[i::A] if self.centralized_critic else agent_obs
                        
                        act, comm_act, logprob, _, val = self.agents[agent].get_action_and_value(agent_obs, agent_gobs)
                        
                        buffers[agent]["obs"][step] = agent_obs
                        buffers[agent]["global_obs"][step] = agent_gobs
                        buffers[agent]["actions"][step] = act
                        buffers[agent]["logprobs"][step] = logprob
                        buffers[agent]["values"][step] = val.squeeze()
                        if self.comm_enabled:
                            buffers[agent]["comm_actions"][step] = comm_act
                            
                        actions_to_env[i::A] = act.cpu().numpy()
                        if self.comm_enabled:
                            comm_actions_to_env[i::A] = comm_act.cpu().numpy()

                if self.comm_enabled:
                    step_action = {"action": actions_to_env, "message": comm_actions_to_env}
                else:
                    step_action = actions_to_env
                    
                next_obs, rews, terms, truncs, infos = self.env.step(step_action)
                
                scale = float(self.ablation.get("reward_scale", self.config.get("reward_scale", 100.0)))
                rews_scaled = torch.FloatTensor(rews).to(self.device) / scale
                terms_tensor = torch.FloatTensor(terms).to(self.device)
                
                total_env_reward += rews.sum()
                
                for i, agent in enumerate(self.agents_names):
                    buffers[agent]["rewards"][step] = rews_scaled[i::A]
                    buffers[agent]["terminations"][step] = terms_tensor[i::A]
                    
                self.next_obs = next_obs
            
            # (Advantages and PPO Update logic omitted for brevity in CI smoke test, 
            #  but realistically we should just do a dummy update or standard PPO update if we want real training).
            # I will include a basic GAE and PPO update to ensure it actually trains!
            
            # --- PPO Update ---
            obs_tensor = self.flatten_obs(self.next_obs)
            global_obs_tensor = self._get_global_obs(obs_tensor)
            
            for i, agent in enumerate(self.agents_names):
                agent_obs = obs_tensor[i::A]
                agent_gobs = global_obs_tensor[i::A] if self.centralized_critic else agent_obs
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
                
                # Advantages logic remains the same up to here.
                # Just before 'Update loop for this agent'
                # We need to reshape buffers for minibatches.
                b_obs = buffers[agent]["obs"].reshape(-1, self.obs_dim)
                b_gobs = buffers[agent]["global_obs"].reshape(-1, self.global_obs_dim if self.centralized_critic else self.obs_dim)
                b_actions = buffers[agent]["actions"].reshape(-1)
                b_logprobs = buffers[agent]["logprobs"].reshape(-1)
                b_advantages = advantages.reshape(-1)
                b_returns = returns.reshape(-1)
                b_comm = buffers[agent]["comm_actions"].reshape(-1, 4) if self.comm_enabled else None
                
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
                b_inds = np.arange(len(b_obs))
                
                opt = self.optimizers[agent] if not self.shared_params else self.optimizers[self.agents_names[0]]
                opt_agent = agent if not self.shared_params else self.agents_names[0]
                
                epochs = self.config.get("epochs", 10)
                minibatch_size = self.config.get("minibatch_size", 64)
                
                for epoch in range(epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, len(b_obs), minibatch_size):
                        end = start + minibatch_size
                        mb_inds = b_inds[start:end]
                        
                        mb_comm = b_comm[mb_inds] if self.comm_enabled else None
                        _, _, newlogprob, entropy, newvalue = self.agents[opt_agent].get_action_and_value(b_obs[mb_inds], b_gobs[mb_inds], b_actions[mb_inds], mb_comm)
                        
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()
                        
                        mb_advantages = b_advantages[mb_inds]
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config["clip_epsilon"], 1 + self.config["clip_epsilon"])
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                        
                        v_loss = 0.5 * ((newvalue.squeeze() - b_returns[mb_inds]) ** 2).mean()
                        
                        loss = pg_loss - self.config["entropy_coef"] * entropy.mean() + v_loss * self.config["value_coef"]
                        
                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.agents[opt_agent].parameters(), self.config["max_grad_norm"])
                        opt.step()
                
                if agent == "retailer":
                    metrics_sum = {
                        "retailer_policy_loss": pg_loss.item(),
                        "retailer_value_loss": v_loss.item(),
                        "retailer_reward": total_env_reward / N
                    }
                    
        return metrics_sum
