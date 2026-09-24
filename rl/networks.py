import torch
from torch import nn
from torch.distributions.categorical import Categorical


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.action_dim = action_dim
        self.register_buffer("bins", torch.arange(action_dim).float())
        import math

        self.log_std = nn.Parameter(torch.full((1,), math.log(20.0)))

    def forward(self, obs: torch.Tensor) -> Categorical:
        # Scale observations to prevent Tanh saturation
        # Scale the output by 100.0 so the network only needs to learn weights in [-1, 1]
        mean = self.net(obs / 100.0) * 100.0  # shape (B, 1)
        std = torch.clamp(self.log_std.exp(), min=1.0)
        logits = -0.5 * ((self.bins.unsqueeze(0) - mean) / std) ** 2
        return Categorical(logits=logits)


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Scale observations to prevent Tanh saturation
        return self.net(obs / 100.0).squeeze(-1)
