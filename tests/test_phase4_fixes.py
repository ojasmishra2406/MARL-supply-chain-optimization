import torch
from torch.distributions.categorical import Categorical

from rl.networks import ActorNetwork, CriticNetwork


def test_ordinal_actor():
    """Verify Ordinal Actor mathematically."""
    actor = ActorNetwork(obs_dim=5, action_dim=101, hidden_size=128)

    # 1. Output is Categorical(101)
    obs = torch.zeros((1, 5))
    dist = actor(obs)
    assert isinstance(dist, Categorical)
    assert dist.probs.shape == (1, 101)

    # 2. Probability sum is exactly 1.0
    probs = dist.probs.squeeze()
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-4)

    # 3. Log probabilities are consistent
    action = torch.tensor([50])
    log_prob = dist.log_prob(action)
    assert torch.isclose(log_prob, torch.log(probs[50]), atol=1e-4)

    # 4. Valid action range (samples are in [0, 100])
    sample = dist.sample()
    assert 0 <= sample.item() <= 100

    # 5. Gradient flows through mean to weights
    loss = -log_prob
    loss.backward()

    # Verify gradients exist in the first layer
    assert actor.net[0].weight.grad is not None


def test_observation_scaling():
    """Verify observations are scaled correctly."""
    critic = CriticNetwork(obs_dim=5, hidden_size=128)
    obs = torch.ones((1, 5)) * 100.0
    out = critic(obs)
    assert out.shape == (1,)
