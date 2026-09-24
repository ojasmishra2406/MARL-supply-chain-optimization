import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from rl.gnn_network import GNN_MAPPO_Critic, GNN_MAPPO_Actor

class Phase10Agent(nn.Module):
    def __init__(self, obs_dim, action_dim, num_nodes=4, hidden_dim=64, agent_index=0, lr=3e-4):
        super().__init__()
        self.actor = GNN_MAPPO_Actor(num_nodes, obs_dim, action_dim, hidden_dim, agent_index)
        # Note: Centralized critic is managed by the Trainer in MAPPO, 
        # but in our Phase 4 architecture, each agent has its own critic network.
        # We will use the GNN critic here.
        self.critic = GNN_MAPPO_Critic(num_nodes, obs_dim, hidden_dim)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def get_action_and_value(self, obs, global_obs, action=None, comm_action=None):
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(global_obs)
        
        # Match Phase5Agent API: return action, comm_action (None), logprob, entropy, value
        return action, None, log_prob, entropy, value
        
    def get_value(self, obs, global_obs):
        return self.critic(global_obs)
        
    def evaluate_actions(self, obs, global_obs, actions):
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        # Critic evaluates global graph state
        values = self.critic(global_obs).squeeze(-1)
        
        return values, log_probs, entropy
