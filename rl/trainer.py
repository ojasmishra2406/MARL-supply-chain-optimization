import os
import time

import numpy as np
import torch
import wandb
import yaml
from torch import nn, optim

from rl.gae import compute_gae
from rl.ppo import IPPOAgent
from rl.rollout import SyncVectorEnv


class IPPOTrainer:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = self.config.get("seeds", [42])[0]

        # Set determinism
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sim_config = os.path.join(repo_root, self.config["simulator_config"])

        self.num_envs = self.config["num_envs"]
        self.vec_env = SyncVectorEnv(self.num_envs, sim_config)
        self.agents_names = self.vec_env.agents

        # Determine observation and action dimensions
        # From Phase 2, obs is Box(5,), action is Discrete(101)
        # We will instantiate agents independently
        obs_dim = self.vec_env.envs[0].observation_space(self.agents_names[0]).shape[0]
        # The true capacity is in vec_env.envs[0].capacity, so action_dim = capacity + 1
        action_dim = self.vec_env.envs[0].capacity + 1

        hidden_size = self.config["hidden_size"]

        self.agents = nn.ModuleDict(
            {
                agent: IPPOAgent(obs_dim, action_dim, hidden_size).to(self.device)
                for agent in self.agents_names
            }
        )

        lr = self.config["learning_rate"]
        self.optimizers = {
            agent: optim.Adam(self.agents[agent].parameters(), lr=lr)
            for agent in self.agents_names
        }

    def save_checkpoint(self, path: str, update: int):
        checkpoint = {
            "update": update,
            "config": self.config,
            "seed": self.seed,
            "actors": {a: self.agents[a].actor.state_dict() for a in self.agents_names},
            "critics": {
                a: self.agents[a].critic.state_dict() for a in self.agents_names
            },
            "optimizers": {
                a: self.optimizers[a].state_dict() for a in self.agents_names
            },
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        for a in self.agents_names:
            self.agents[a].actor.load_state_dict(checkpoint["actors"][a])
            self.agents[a].critic.load_state_dict(checkpoint["critics"][a])
            self.optimizers[a].load_state_dict(checkpoint["optimizers"][a])
        return checkpoint["update"]

    def train(self, total_updates: int, use_wandb: bool = False):
        if use_wandb:
            wandb.init(
                project="marl-supply-chain",
                config=self.config,
                name=f"ippo_seed_{self.seed}",
            )

        rollout_length = self.config["rollout_length"]
        gamma = self.config["gamma"]
        gae_lambda = self.config["gae_lambda"]
        clip_epsilon = self.config["clip_epsilon"]
        entropy_coef = self.config["entropy_coef"]
        value_coef = self.config["value_coef"]
        max_grad_norm = self.config["max_grad_norm"]
        epochs = self.config["epochs"]
        minibatch_size = self.config["minibatch_size"]

        obs_list = self.vec_env.reset(seed=self.seed)

        # Rollout buffers per agent
        buffers = {
            agent: {
                "obs": torch.zeros(
                    (rollout_length, self.num_envs, self.agents[agent].actor.net[0].in_features), device=self.device
                ),
                "actions": torch.zeros(
                    (rollout_length, self.num_envs),
                    dtype=torch.long,
                    device=self.device,
                ),
                "logprobs": torch.zeros(
                    (rollout_length, self.num_envs), device=self.device
                ),
                "rewards": torch.zeros(
                    (rollout_length, self.num_envs), device=self.device
                ),
                "values": torch.zeros(
                    (rollout_length, self.num_envs), device=self.device
                ),
                "terminations": torch.zeros(
                    (rollout_length, self.num_envs), device=self.device
                ),
                "truncations": torch.zeros(
                    (rollout_length, self.num_envs), device=self.device
                ),
            }
            for agent in self.agents_names
        }

        next_obs = {
            agent: torch.tensor(
                np.stack([o[agent] for o in obs_list]),
                dtype=torch.float32,
                device=self.device,
            )
            for agent in self.agents_names
        }
        next_done = {
            agent: torch.zeros(self.num_envs, device=self.device)
            for agent in self.agents_names
        }

        best_cost = float("inf")

        for update in range(1, total_updates + 1):
            start_time = time.time()

            # Rollout phase
            for step in range(rollout_length):
                actions_to_env = [{} for _ in range(self.num_envs)]

                with torch.no_grad():
                    for agent in self.agents_names:
                        buffers[agent]["obs"][step] = next_obs[agent]
                        action, logprob, _, value = self.agents[
                            agent
                        ].get_action_and_value(next_obs[agent])
                        buffers[agent]["actions"][step] = action
                        buffers[agent]["logprobs"][step] = logprob
                        buffers[agent]["values"][step] = value

                        action_np = action.cpu().numpy()
                        for env_idx in range(self.num_envs):
                            actions_to_env[env_idx][agent] = action_np[env_idx]

                obs_list, rews_list, terms_list, truncs_list, infos_list = (
                    self.vec_env.step(actions_to_env)
                )

                for agent in self.agents_names:
                    # If truncated, true next_obs is in final_observation
                    next_obs_arr = []
                    for i in range(self.num_envs):
                        if truncs_list[i][agent] or terms_list[i][agent]:
                            next_obs_arr.append(
                                infos_list[i]["final_observation"][agent]
                            )
                        else:
                            next_obs_arr.append(obs_list[i][agent])

                    next_obs[agent] = torch.tensor(
                        np.stack(next_obs_arr), dtype=torch.float32, device=self.device
                    )
                    # Scale rewards to stabilize value network
                    scale_factor = self.config.get("reward_scale", 100.0)
                    buffers[agent]["rewards"][step] = torch.tensor(
                        [r[agent] / scale_factor for r in rews_list], device=self.device
                    )
                    # Notice: terminations here are treated as the end of the episode for GAE
                    buffers[agent]["terminations"][step] = torch.tensor(
                        [
                            float(t[agent] or truncs_list[i][agent])
                            for i, t in enumerate(terms_list)
                        ],
                        device=self.device,
                    )
                    next_done[agent] = torch.tensor(
                        [
                            float(t[agent] or truncs_list[i][agent])
                            for i, t in enumerate(terms_list)
                        ],
                        device=self.device,
                    )

            # Optimization phase per agent
            metrics = {}
            for agent in self.agents_names:
                with torch.no_grad():
                    next_value = self.agents[agent].get_value(next_obs[agent])
                    advantages = compute_gae(
                        buffers[agent]["rewards"],
                        buffers[agent]["values"],
                        buffers[agent]["terminations"],
                        next_value,
                        next_done[agent],
                        gamma,
                        gae_lambda,
                    )
                    returns = advantages + buffers[agent]["values"]
                    buffers[agent]["returns"] = returns
                    buffers[agent]["advantages"] = advantages

                obs_dim = self.agents[agent].actor.net[0].in_features
                b_obs = buffers[agent]["obs"].reshape(-1, obs_dim)
                b_logprobs = buffers[agent]["logprobs"].reshape(-1)
                b_actions = buffers[agent]["actions"].reshape(-1)
                b_advantages = advantages.reshape(-1)

                # GLOBAL ADVANTAGE NORMALIZATION (Fixing per-minibatch normalization bug)
                b_advantages = (b_advantages - b_advantages.mean()) / (
                    b_advantages.std() + 1e-8
                )

                b_returns = returns.reshape(-1)
                buffers[agent]["values"].reshape(-1)

                b_inds = np.arange(self.num_envs * rollout_length)

                clipfracs = []
                for epoch in range(epochs):
                    np.random.shuffle(b_inds)
                    for start in range(
                        0, self.num_envs * rollout_length, minibatch_size
                    ):
                        end = start + minibatch_size
                        mb_inds = b_inds[start:end]

                        _, newlogprob, entropy, newvalue = self.agents[
                            agent
                        ].get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            clipfracs += [
                                ((ratio - 1.0).abs() > clip_epsilon)
                                .float()
                                .mean()
                                .item()
                            ]

                        mb_advantages = b_advantages[mb_inds]

                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(
                            ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon
                        )
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                        entropy_loss = entropy.mean()

                        loss = (
                            pg_loss - entropy_coef * entropy_loss + value_coef * v_loss
                        )

                        self.optimizers[agent].zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(
                            self.agents[agent].parameters(), max_grad_norm
                        )
                        self.optimizers[agent].step()

                # Rewards were scaled down by scale_factor, so scale them back up for logging
                scale_factor = self.config.get("reward_scale", 100.0)
                metrics[f"{agent}_reward"] = (
                    buffers[agent]["rewards"].sum(0).mean().item() * scale_factor
                )
                metrics[f"{agent}_loss"] = loss.item()
                metrics[f"{agent}_policy_loss"] = pg_loss.item()
                metrics[f"{agent}_value_loss"] = v_loss.item()
                metrics[f"{agent}_entropy"] = entropy_loss.item()
                
                # Explained variance and Advantage variance over the full buffer
                y_pred = buffers[agent]["values"].cpu().numpy()
                y_true = buffers[agent]["returns"].cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                
                adv_var = np.var(buffers[agent]["advantages"].cpu().numpy())
                
                metrics[f"{agent}_explained_var"] = explained_var
                metrics[f"{agent}_adv_var"] = adv_var

            total_cost = -sum(metrics[f"{a}_reward"] for a in self.agents_names)
            best_cost = min(best_cost, total_cost)

            metrics["total_cost"] = total_cost
            metrics["best_cost"] = best_cost
            metrics["time"] = time.time() - start_time

            if use_wandb:
                wandb.log(metrics, step=update)

        if use_wandb:
            wandb.finish()

        return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase4_ippo.yaml")
    args = parser.parse_args()

    trainer = IPPOTrainer(args.config)
    trainer.train(total_updates=2)
