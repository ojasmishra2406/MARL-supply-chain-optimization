import os
import time
import yaml
import torch
import torch.nn as nn
import numpy as np
import wandb

from rl.gae import compute_gae
from rl.rollout import SyncVectorEnv
from rl.networks import ActorNetwork, CriticNetwork
from torch.distributions.normal import Normal

def flatten_obs(obs):
    if isinstance(obs, dict):
        return np.concatenate([obs["state"], obs["message_upstream"], obs["message_downstream"]])
    return obs

class Phase5Agent(nn.Module):
    def __init__(self, obs_dim, global_obs_dim, action_dim, hidden_size=128, comm_dim=0, centralized_critic=True):
        super().__init__()
        self.comm_dim = comm_dim
        self.centralized_critic = centralized_critic
        
        self.actor = ActorNetwork(obs_dim, action_dim, hidden_size)
        if comm_dim > 0:
            self.comm_net = nn.Sequential(
                nn.Linear(obs_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, comm_dim)
            )
            self.comm_log_std = nn.Parameter(torch.zeros(comm_dim))
            
        critic_dim = global_obs_dim if centralized_critic else obs_dim
        self.critic = CriticNetwork(critic_dim, hidden_size)
        
    def get_action_and_value(self, obs, global_obs, action=None, comm_action=None):
        action_dist = self.actor(obs)
        if action is None:
            action = action_dist.sample()
        action_logprob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        
        if self.comm_dim > 0:
            comm_mean = self.comm_net(obs / 100.0)
            comm_std = torch.clamp(self.comm_log_std.exp(), min=1e-3, max=10.0)
            comm_dist = Normal(comm_mean, comm_std)
            if comm_action is None:
                comm_action = comm_dist.sample()
            comm_logprob = comm_dist.log_prob(comm_action).sum(dim=-1)
            entropy += comm_dist.entropy().sum(dim=-1)
            action_logprob += comm_logprob
            
        critic_input = global_obs if self.centralized_critic else obs
        value = self.critic(critic_input)
        
        return action, comm_action, action_logprob, entropy, value

    def get_value(self, obs, global_obs):
        critic_input = global_obs if self.centralized_critic else obs
        return self.critic(critic_input)

class Phase5Trainer:
    def __init__(self, config_path: str, ablation_config: dict):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.ablation = ablation_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = self.ablation.get("seed", 0)
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sim_config_path = os.path.join(repo_root, self.config["simulator_config"])
        
        with open(sim_config_path, "r") as f:
            sim_cfg = yaml.safe_load(f)
            
        self.comm_enabled = self.ablation.get("communication", False)
        sim_cfg["communication_enabled"] = self.comm_enabled
        
        if "reward_weights" in self.ablation:
            sim_cfg["cost"] = self.ablation["reward_weights"]
            
        os.makedirs("configs", exist_ok=True)
        temp_sim_config = f"configs/temp_phase5_sim_{self.seed}.yaml"
        with open(temp_sim_config, "w") as f:
            yaml.dump(sim_cfg, f)
            
        self.num_envs = self.config["num_envs"]
        self.vec_env = SyncVectorEnv(self.num_envs, temp_sim_config, comm_enabled=self.comm_enabled)
        self.agents_names = self.vec_env.agents
        
        sample_obs = self.vec_env.envs[0].reset()[0][self.agents_names[0]]
        obs_dim = len(flatten_obs(sample_obs))
        global_obs_dim = obs_dim * len(self.agents_names)
        action_dim = self.vec_env.envs[0].capacity + 1
        self.comm_dim = 4 if self.comm_enabled else 0
        
        hidden_size = self.config["hidden_size"]
        self.shared_params = self.ablation.get("parameter_sharing", False)
        self.centralized_critic = self.ablation.get("centralized_critic", True)
        
        if self.shared_params:
            shared_agent = Phase5Agent(obs_dim, global_obs_dim, action_dim, hidden_size, self.comm_dim, self.centralized_critic).to(self.device)
            self.agents = nn.ModuleDict({a: shared_agent for a in self.agents_names})
            lr = self.ablation.get("learning_rate", self.config["learning_rate"])
            self.optimizers = {"shared": torch.optim.Adam(shared_agent.parameters(), lr=lr, eps=1e-5)}
        else:
            self.agents = nn.ModuleDict({
                a: Phase5Agent(obs_dim, global_obs_dim, action_dim, hidden_size, self.comm_dim, self.centralized_critic).to(self.device)
                for a in self.agents_names
            })
            lr = self.ablation.get("learning_rate", self.config["learning_rate"])
            self.optimizers = {
                a: torch.optim.Adam(self.agents[a].parameters(), lr=lr, eps=1e-5)
                for a in self.agents_names
            }

    def train(self, total_updates=1, use_wandb=False):
        rollout_length = self.config["rollout_length"]
        gamma = self.config["gamma"]
        gae_lambda = self.config["gae_lambda"]
        clip_epsilon = self.config["clip_epsilon"]
        entropy_coef = self.config["entropy_coef"]
        value_coef = self.config["value_coef"]
        max_grad_norm = self.config["max_grad_norm"]
        epochs = self.config["epochs"]
        minibatch_size = self.config["minibatch_size"]
        
        buffers = {
            a: {
                "obs": torch.zeros((rollout_length, self.num_envs, self.agents[a].actor.net[0].in_features), device=self.device),
                "global_obs": torch.zeros((rollout_length, self.num_envs, self.agents[a].critic.net[0].in_features), device=self.device),
                "actions": torch.zeros((rollout_length, self.num_envs), dtype=torch.long, device=self.device),
                "logprobs": torch.zeros((rollout_length, self.num_envs), device=self.device),
                "rewards": torch.zeros((rollout_length, self.num_envs), device=self.device),
                "values": torch.zeros((rollout_length, self.num_envs), device=self.device),
                "terminations": torch.zeros((rollout_length, self.num_envs), device=self.device),
            }
            for a in self.agents_names
        }
        if self.comm_enabled:
            for a in self.agents_names:
                buffers[a]["comm_actions"] = torch.zeros((rollout_length, self.num_envs, self.comm_dim), device=self.device)
                
        next_obs_dict = self.vec_env.reset(seed=self.seed)
        
        def build_tensors(obs_list):
            obs_tensors = {}
            for agent in self.agents_names:
                arr = [flatten_obs(env_obs[agent]) for env_obs in obs_list]
                obs_tensors[agent] = torch.tensor(np.stack(arr), dtype=torch.float32, device=self.device)
            global_obs = torch.cat([obs_tensors[a] for a in self.agents_names], dim=-1)
            return obs_tensors, global_obs
            
        next_obs, next_global_obs = build_tensors(next_obs_dict)
        next_done = {a: torch.zeros(self.num_envs, device=self.device) for a in self.agents_names}
        
        final_metrics = {}
        for update in range(1, total_updates + 1):
            for step in range(rollout_length):
                actions_to_env = [{} for _ in range(self.num_envs)]
                with torch.no_grad():
                    for agent in self.agents_names:
                        buffers[agent]["obs"][step] = next_obs[agent]
                        buffers[agent]["global_obs"][step] = next_global_obs if self.centralized_critic else next_obs[agent]
                        
                        act, comm_act, logprob, _, val = self.agents[agent].get_action_and_value(
                            next_obs[agent], next_global_obs if self.centralized_critic else next_obs[agent]
                        )
                        buffers[agent]["actions"][step] = act
                        if self.comm_enabled:
                            buffers[agent]["comm_actions"][step] = comm_act
                        buffers[agent]["logprobs"][step] = logprob
                        buffers[agent]["values"][step] = val
                        
                        act_np = act.cpu().numpy()
                        if self.comm_enabled:
                            comm_np = comm_act.cpu().numpy()
                        for env_idx in range(self.num_envs):
                            if self.comm_enabled:
                                actions_to_env[env_idx][agent] = {"action": act_np[env_idx], "message": comm_np[env_idx]}
                            else:
                                actions_to_env[env_idx][agent] = act_np[env_idx]
                                
                obs_list, rews_list, terms_list, truncs_list, infos_list = self.vec_env.step(actions_to_env)
                
                real_obs_list = []
                for i in range(self.num_envs):
                    if any(terms_list[i].values()) or any(truncs_list[i].values()):
                        real_obs_list.append(infos_list[i]["final_observation"])
                    else:
                        real_obs_list.append(obs_list[i])
                        
                next_obs, next_global_obs = build_tensors(real_obs_list)
                
                scale_factor = self.ablation.get("reward_scale", self.config.get("reward_scale", 100.0))
                for agent in self.agents_names:
                    buffers[agent]["rewards"][step] = torch.tensor(
                        [r[agent] / scale_factor for r in rews_list], device=self.device
                    )
                    buffers[agent]["terminations"][step] = torch.tensor(
                        [float(t[agent] or truncs_list[i][agent]) for i, t in enumerate(terms_list)], device=self.device
                    )
                    next_done[agent] = buffers[agent]["terminations"][step].clone()
                    
            for agent in self.agents_names:
                with torch.no_grad():
                    next_val = self.agents[agent].get_value(next_obs[agent], next_global_obs if self.centralized_critic else next_obs[agent])
                    advantages = compute_gae(
                        buffers[agent]["rewards"],
                        buffers[agent]["values"],
                        buffers[agent]["terminations"],
                        next_val,
                        next_done[agent],
                        gamma,
                        gae_lambda,
                    )
                    buffers[agent]["returns"] = advantages + buffers[agent]["values"]
                    buffers[agent]["advantages"] = advantages
                    
            # Optimization
            if self.shared_params:
                def get_merged(key):
                    return torch.cat([buffers[a][key].reshape(-1, *buffers[a][key].shape[2:]) for a in self.agents_names], dim=0)
                
                b_obs = get_merged("obs")
                b_gobs = get_merged("global_obs")
                b_actions = get_merged("actions")
                if self.comm_enabled:
                    b_comm = get_merged("comm_actions")
                b_logprobs = get_merged("logprobs")
                b_advs = get_merged("advantages")
                b_rets = get_merged("returns")
                
                b_advs = (b_advs - b_advs.mean()) / (b_advs.std() + 1e-8)
                b_inds = np.arange(len(b_obs))
                
                for epoch in range(epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, len(b_obs), minibatch_size):
                        end = start + minibatch_size
                        mb_inds = b_inds[start:end]
                        
                        _, _, newlogprob, entropy, newval = self.agents[self.agents_names[0]].get_action_and_value(
                            b_obs[mb_inds], b_gobs[mb_inds], b_actions[mb_inds], 
                            b_comm[mb_inds] if self.comm_enabled else None
                        )
                        
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()
                        
                        mb_adv = b_advs[mb_inds]
                        pg_loss1 = -mb_adv * ratio
                        pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                        v_loss = 0.5 * ((newval - b_rets[mb_inds])**2).mean()
                        entropy_loss = entropy.mean()
                        
                        loss = pg_loss - entropy_coef * entropy_loss + value_coef * v_loss
                        
                        self.optimizers["shared"].zero_grad()
                        loss.backward()
                        grad_norm = nn.utils.clip_grad_norm_(self.agents[self.agents_names[0]].parameters(), max_grad_norm)
                        self.optimizers["shared"].step()
            else:
                for agent in self.agents_names:
                    b_obs = buffers[agent]["obs"].reshape(-1, buffers[agent]["obs"].shape[-1])
                    b_gobs = buffers[agent]["global_obs"].reshape(-1, buffers[agent]["global_obs"].shape[-1])
                    b_actions = buffers[agent]["actions"].reshape(-1)
                    if self.comm_enabled:
                        b_comm = buffers[agent]["comm_actions"].reshape(-1, self.comm_dim)
                    b_logprobs = buffers[agent]["logprobs"].reshape(-1)
                    b_advs = buffers[agent]["advantages"].reshape(-1)
                    b_rets = buffers[agent]["returns"].reshape(-1)
                    
                    b_advs = (b_advs - b_advs.mean()) / (b_advs.std() + 1e-8)
                    b_inds = np.arange(len(b_obs))
                    
                    grad_norms = []
                    for epoch in range(epochs):
                        np.random.shuffle(b_inds)
                        for start in range(0, len(b_obs), minibatch_size):
                            end = start + minibatch_size
                            mb_inds = b_inds[start:end]
                            
                            _, _, newlogprob, entropy, newval = self.agents[agent].get_action_and_value(
                                b_obs[mb_inds], b_gobs[mb_inds], b_actions[mb_inds],
                                b_comm[mb_inds] if self.comm_enabled else None
                            )
                            
                            logratio = newlogprob - b_logprobs[mb_inds]
                            ratio = logratio.exp()
                            
                            mb_adv = b_advs[mb_inds]
                            pg_loss1 = -mb_adv * ratio
                            pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
                            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                            v_loss = 0.5 * ((newval - b_rets[mb_inds])**2).mean()
                            entropy_loss = entropy.mean()
                            
                            loss = pg_loss - entropy_coef * entropy_loss + value_coef * v_loss
                            
                            self.optimizers[agent].zero_grad()
                            loss.backward()
                            grad_norm = nn.utils.clip_grad_norm_(self.agents[agent].parameters(), max_grad_norm)
                            grad_norms.append(grad_norm.item())
                            self.optimizers[agent].step()
                            
            # Metrics
            metrics = {}
            for agent in self.agents_names:
                scale_factor = self.ablation.get("reward_scale", self.config.get("reward_scale", 100.0))
                metrics[f"{agent}_reward"] = buffers[agent]["rewards"].sum(0).mean().item() * scale_factor
                y_pred = buffers[agent]["values"].cpu().numpy()
                y_true = buffers[agent]["returns"].cpu().numpy()
                var_y = np.var(y_true)
                ev = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                metrics[f"{agent}_explained_var"] = float(ev)
                metrics[f"{agent}_entropy"] = entropy_loss.item()
                metrics[f"{agent}_policy_loss"] = pg_loss.item()
                metrics[f"{agent}_value_loss"] = v_loss.item()
                metrics[f"{agent}_grad_norm"] = float(np.mean(grad_norms)) if not self.shared_params else float(grad_norm.item())
                metrics[f"{agent}_clip_frac"] = ((torch.exp(newlogprob - b_logprobs[mb_inds]) - 1.0).abs() > clip_epsilon).float().mean().item()

            metrics["total_cost"] = -sum(metrics[f"{a}_reward"] for a in self.agents_names)
            final_metrics = metrics
            
        return final_metrics
