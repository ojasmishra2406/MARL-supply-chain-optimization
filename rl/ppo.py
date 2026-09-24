import torch
from torch import nn

from rl.networks import ActorNetwork, CriticNetwork


class IPPOAgent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.actor = ActorNetwork(obs_dim, action_dim, hidden_size)
        self.critic = CriticNetwork(obs_dim, hidden_size)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None):
        dist = self.actor(obs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(obs)
        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor):
        return self.critic(obs)


class MAPPOAgent(nn.Module):
    def __init__(self, obs_dim: int, global_obs_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.actor = ActorNetwork(obs_dim, action_dim, hidden_size)
        self.critic = CriticNetwork(global_obs_dim, hidden_size)

    def get_action(self, obs: torch.Tensor, action: torch.Tensor = None):
        dist = self.actor(obs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def get_value(self, global_obs: torch.Tensor):
        return self.critic(global_obs)

